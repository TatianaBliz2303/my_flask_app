from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import logging
import os
from datetime import datetime, timedelta
import urllib3
import base64
import uuid
import threading
import time
from queue import Queue
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Отключаем предупреждения о небезопасных запросах
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Глобальные переменные для хранения данных
embeddings = []
text = ""
access_token = None
token_expiry = None
embeddings_queue = Queue()
processing_complete = False

# Константы из переменных окружения
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1"
CLIENT_ID = os.getenv('CLIENT_ID')
AUTH_KEY = os.getenv('AUTH_KEY')
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Путь к файлу для сохранения эмбеддингов
EMBEDDINGS_FILE = "embeddings.json"

def get_access_token():
    """Получение токена доступа от GigaChat API"""
    global access_token, token_expiry
    
    # Проверяем, не истек ли текущий токен
    if access_token and token_expiry and datetime.now() < token_expiry:
        return access_token
    
    try:
        # Запрос токена
        headers = {
            "Authorization": f"Bearer {AUTH_KEY}",
            "RqUID": CLIENT_ID,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = "scope=GIGACHAT_API_PERS"
        
        response = requests.post(
            AUTH_URL, 
            headers=headers, 
            data=data, 
            verify=False
        )
        
        if response.status_code == 401:
            logger.error("Неверные учетные данные")
            raise Exception("Неверные учетные данные")
        
        response.raise_for_status()
        data = response.json()
        
        access_token = data["access_token"]
        token_expiry = datetime.now() + timedelta(seconds=3600 - 60)  # Токен живет 1 час
        logger.info("Токен успешно получен")
        
        return access_token
    except Exception as e:
        logger.error(f"Ошибка при получении токена: {str(e)}")
        raise

def load_book_from_file(file_path="book.txt"):
    """Загрузка книги из файла"""
    global text
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        
        logger.info(f"Книга успешно загружена из {file_path}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при загрузке книги: {str(e)}")
        return False

def save_embeddings():
    """Сохранение эмбеддингов в файл"""
    try:
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f)
        logger.info(f"Эмбеддинги успешно сохранены в файл {EMBEDDINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении эмбеддингов: {str(e)}")
        return False

def load_embeddings():
    """Загрузка эмбеддингов из файла"""
    global embeddings
    try:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
            logger.info(f"Эмбеддинги успешно загружены из файла {EMBEDDINGS_FILE}")
            return True
        return False
    except Exception as e:
        logger.error(f"Ошибка при загрузке эмбеддингов: {str(e)}")
        return False

def create_embeddings_with_gigachat(text_chunks):
    """Создание эмбеддингов с помощью GigaChat API"""
    global embeddings, processing_complete
    
    try:
        token = get_access_token()
        
        for i, chunk in enumerate(text_chunks, 1):
            logger.info(f"Обработка чанка {i}/{len(text_chunks)}")
            
            # Добавляем задержку между запросами, чтобы избежать ошибки 429
            if i > 1:
                time.sleep(1)  # Задержка 1 секунда между запросами
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{GIGACHAT_API_URL}/embeddings",
                headers=headers,
                json={
                    "model": "Embeddings",
                    "input": chunk,
                    "encoding_type": "float"
                },
                verify=False
            )
            
            if response.status_code == 401:
                # Если токен истек, получаем новый
                token = get_access_token()
                response = requests.post(
                    f"{GIGACHAT_API_URL}/embeddings",
                    headers={**headers, "Authorization": f"Bearer {token}"},
                    json={
                        "model": "Embeddings",
                        "input": chunk,
                        "encoding_type": "float"
                    },
                    verify=False
                )
            
            if response.status_code == 429:
                # Если превышен лимит запросов, ждем 5 секунд и пробуем снова
                logger.warning("Превышен лимит запросов. Ожидание 5 секунд...")
                time.sleep(5)
                i -= 1  # Повторяем этот чанк
                continue
                
            response.raise_for_status()
            data = response.json()
            
            embeddings.append(data["data"][0]["embedding"])
            logger.info(f"Успешно создан эмбеддинг для чанка {i}")
            
            # Сохраняем эмбеддинги каждые 10 чанков
            if i % 10 == 0:
                save_embeddings()
        
        logger.info("Эмбеддинги успешно созданы")
        save_embeddings()  # Сохраняем финальный результат
        return True
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддингов: {str(e)}")
        return False

def initialize_book():
    """Инициализация книги и создание эмбеддингов при запуске сервера"""
    global processing_complete
    
    if not load_book_from_file():
        logger.error("Не удалось загрузить книгу")
        return False
    
    # Пробуем загрузить существующие эмбеддинги
    if load_embeddings():
        logger.info("Используем существующие эмбеддинги")
        processing_complete = True
        return True
    
    # Если эмбеддинги не найдены, создаем новые
    logger.info("Существующие эмбеддинги не найдены, создаем новые")
    
    # Разбиваем текст на чанки по 150 символов
    chunks = [text[i:i+150] for i in range(0, len(text), 150)]
    
    # Ограничиваем количество чанков для быстрого запуска
    # Полностью обработаем только первые 50 чанков
    initial_chunks = chunks[:50]
    logger.info(f"Запуск сервера с обработкой {len(initial_chunks)} из {len(chunks)} чанков")
    
    if not create_embeddings_with_gigachat(initial_chunks):
        logger.error("Не удалось создать эмбеддинги")
        return False
    
    # Запускаем фоновую задачу для обработки остальных чанков
    threading.Thread(target=process_remaining_chunks, args=(chunks[50:],), daemon=True).start()
    
    return True

def process_remaining_chunks(remaining_chunks):
    """Обработка оставшихся чанков в фоновом режиме"""
    global processing_complete
    
    if not remaining_chunks:
        processing_complete = True
        return
    
    logger.info(f"Начало фоновой обработки {len(remaining_chunks)} оставшихся чанков")
    try:
        # Обрабатываем чанки небольшими партиями, чтобы избежать ошибки 429
        batch_size = 20
        for i in range(0, len(remaining_chunks), batch_size):
            batch = remaining_chunks[i:i+batch_size]
            logger.info(f"Обработка партии {i//batch_size + 1}/{(len(remaining_chunks) + batch_size - 1)//batch_size}")
            create_embeddings_with_gigachat(batch)
            # Делаем паузу между партиями
            time.sleep(5)
        
        logger.info("Фоновая обработка чанков завершена")
        processing_complete = True
    except Exception as e:
        logger.error(f"Ошибка при фоновой обработке чанков: {str(e)}")
        processing_complete = True

def find_relevant_contexts(question, max_contexts=5):
    """Поиск всех релевантных контекстов в книге"""
    try:
        logger.info("Поиск релевантных контекстов")
        # Создаем эмбеддинг для вопроса
        token = get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{GIGACHAT_API_URL}/embeddings",
            headers=headers,
            json={
                "model": "Embeddings",
                "input": question,
                "encoding_type": "float"
            },
            verify=False
        )
        
        if response.status_code == 401:
            token = get_access_token()
            headers["Authorization"] = f"Bearer {token}"
            response = requests.post(
                f"{GIGACHAT_API_URL}/embeddings",
                headers=headers,
                json={
                    "model": "Embeddings",
                    "input": question,
                    "encoding_type": "float"
                },
                verify=False
            )
        
        response.raise_for_status()
        question_embedding = response.json()["data"][0]["embedding"]
        
        # Находим все релевантные контексты
        similarities = []
        for i, embedding in enumerate(embeddings):
            similarity = sum(a * b for a, b in zip(question_embedding, embedding))
            start_idx = i * 150
            end_idx = min(start_idx + 150, len(text))
            context = text[start_idx:end_idx]
            similarities.append((similarity, context))
        
        # Сортируем по релевантности и берем top-N
        similarities.sort(reverse=True)
        relevant_contexts = [ctx for _, ctx in similarities[:max_contexts]]
        
        # Объединяем контексты
        combined_context = "\n\n".join(relevant_contexts)
        logger.info(f"Найдено {len(relevant_contexts)} релевантных контекстов")
        return combined_context
        
    except Exception as e:
        logger.error(f"Ошибка при поиске релевантных контекстов: {str(e)}")
        return None

@app.route('/')
def index():
    """Отдача главной страницы"""
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_text():
    """Анализ текста и генерация альтернативной истории"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        logger.info("Получен запрос на анализ текста")
        data = request.get_json()
        if not data or 'text' not in data:
            logger.error("Текст не предоставлен в запросе")
            return jsonify({"error": "Текст не предоставлен"}), 400
        
        question = data['text']
        logger.info(f"Вопрос для анализа: {question[:50]}...")
        
        # Проверяем, загружена ли книга и созданы ли эмбеддинги
        if not text or not embeddings:
            logger.error("Книга не загружена или эмбеддинги не созданы")
            return jsonify({"error": "Сервер не готов к обработке запросов. Пожалуйста, подождите."}), 503
        
        # Получаем расширенный контекст
        relevant_context = find_relevant_contexts(question)
        if not relevant_context:
            return jsonify({"error": "Ошибка при поиске контекста"}), 500
        
        # Генерируем альтернативную историю
        try:
            logger.info("Генерация альтернативной истории")
            token = get_access_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{GIGACHAT_API_URL}/chat/completions",
                headers=headers,
                json={
                    "model": "GigaChat",
                    "messages": [
                        {
                            "role": "system", 
                            "content": """Ты - эксперт по альтернативной истории и литературному анализу. 
                            Твоя задача - на основе предоставленного контекста из книги создать правдоподобный 
                            сценарий развития событий. Используй стиль и атмосферу оригинального текста, 
                            сохраняй характерные особенности повествования. Ответ должен быть в том же стиле, 
                            что и книга, с похожими описаниями и атмосферой."""
                        },
                        {
                            "role": "user", 
                            "content": f"""Контекст из книги:
                            {relevant_context}
                            
                            Вопрос: {question}
                            
                            Пожалуйста, опиши альтернативный сценарий развития событий, сохраняя стиль и атмосферу книги."""
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                verify=False
            )
            
            if response.status_code == 401:
                token = get_access_token()
                headers["Authorization"] = f"Bearer {token}"
                response = requests.post(
                    f"{GIGACHAT_API_URL}/chat/completions",
                    headers=headers,
                    json={
                        "model": "GigaChat",
                        "messages": [
                            {
                                "role": "system", 
                                "content": """Ты - эксперт по альтернативной истории и литературному анализу. 
                                Твоя задача - на основе предоставленного контекста из книги создать правдоподобный 
                                сценарий развития событий. Используй стиль и атмосферу оригинального текста, 
                                сохраняй характерные особенности повествования. Ответ должен быть в том же стиле, 
                                что и книга, с похожими описаниями и атмосферой."""
                            },
                            {
                                "role": "user", 
                                "content": f"""Контекст из книги:
                                {relevant_context}
                                
                                Вопрос: {question}
                                
                                Пожалуйста, опиши альтернативный сценарий развития событий, сохраняя стиль и атмосферу книги."""
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    verify=False
                )
            
            response.raise_for_status()
            alternative_history = response.json()["choices"][0]["message"]["content"]
            logger.info("Альтернативная история успешно сгенерирована")
        except Exception as e:
            logger.error(f"Ошибка при генерации альтернативной истории: {str(e)}")
            return jsonify({"error": "Ошибка при генерации ответа"}), 500
        
        return jsonify({
            "question": question,
            "alternative_history": alternative_history
        })
    
    except Exception as e:
        logger.error(f"Необработанная ошибка при анализе текста: {str(e)}")
        return jsonify({"error": f"Внутренняя ошибка сервера: {str(e)}"}), 500

@app.route('/status')
def status():
    """Проверка статуса сервера"""
    return jsonify({
        "status": "ok",
        "book_loaded": bool(text and embeddings),
        "processing_complete": processing_complete
    })

if __name__ == '__main__':
    if initialize_book():
        logger.info("Сервер успешно инициализирован")
        app.run(debug=True)
    else:
        logger.error("Не удалось инициализировать сервер") 