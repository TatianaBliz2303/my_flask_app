import os
import requests
import urllib3
from dotenv import load_dotenv
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

# Отключаем предупреждения о небезопасных запросах
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class GigaChatAPI:
    def __init__(self):
        try:
            self.client_id = os.getenv('CLIENT_ID')
            self.auth_key = os.getenv('AUTH_KEY')
            self.access_token = os.getenv('ACCESS_TOKEN')
            
            if not self.client_id or not self.auth_key:
                raise ValueError("CLIENT_ID или AUTH_KEY не найдены в .env файле")
            
            # Если токен не найден в .env, получаем новый
            if not self.access_token:
                logger.info("Токен не найден в .env файле, получаем новый...")
                self.access_token = self._get_access_token()
                logger.info("Новый токен успешно получен")
        except Exception as e:
            logger.error(f"Ошибка при инициализации GigaChatAPI: {str(e)}")
            raise
    
    def _get_access_token(self):
        """Получение нового токена доступа"""
        try:
            response = requests.post(
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                headers={
                    "Authorization": f"Bearer {self.auth_key}",
                    "RqUID": self.client_id,
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                data={
                    "scope": "GIGACHAT_API_PERS"
                },
                verify=False
            )
            
            if response.status_code != 200:
                raise ValueError(f"Ошибка при получении токена: {response.text}")
            
            return response.json()["access_token"]
        except Exception as e:
            logger.error(f"Ошибка при получении токена: {str(e)}")
            raise
    
    def get_embeddings(self, text):
        """Получение эмбеддингов для текста"""
        try:
            response = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Embeddings",
                    "input": text,
                    "encoding_type": "float"
                },
                verify=False
            )
            
            if response.status_code == 401:
                # Если токен истек, получаем новый и пробуем снова
                logger.info("Токен истек, получаем новый...")
                self.access_token = self._get_access_token()
                return self.get_embeddings(text)
            
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов: {str(e)}")
            raise
    
    def chat_completion(self, messages):
        """Отправка запроса к чат-модели"""
        try:
            response = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "GigaChat",
                    "messages": messages
                },
                verify=False
            )
            
            if response.status_code == 401:
                # Если токен истек, получаем новый и пробуем снова
                logger.info("Токен истек, получаем новый...")
                self.access_token = self._get_access_token()
                return self.chat_completion(messages)
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Ошибка при отправке запроса к чат-модели: {str(e)}")
            raise 