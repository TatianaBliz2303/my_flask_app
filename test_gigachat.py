import requests
import urllib3
import json
from dotenv import load_dotenv
import os

# Загружаем переменные окружения из .env файла
load_dotenv()

# Отключаем предупреждения о небезопасных запросах
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_gigachat_api():
    client_id = os.getenv('CLIENT_ID')
    auth_key = os.getenv('AUTH_KEY')
    access_token = os.getenv('ACCESS_TOKEN')
    
    # Тест получения токена
    print("Testing token retrieval...")
    response = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers={
            "Authorization": f"Bearer {auth_key}",
            "RqUID": client_id,
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "scope": "GIGACHAT_API_PERS"
        },
        verify=False
    )
    print(f"Token response status: {response.status_code}")
    print(f"Token response: {response.text}\n")
    
    if response.status_code == 200:
        # Используем токен из .env файла
        access_token = response.json()["access_token"]
        
        # Тест запроса эмбеддингов
        print("Testing embeddings request...")
        embeddings_response = requests.post(
            "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "Embeddings",
                "input": "This is a test sentence.",
                "encoding_type": "float"
            },
            verify=False
        )
        print(f"Embeddings response status: {embeddings_response.status_code}")
        print(f"Embeddings response: {embeddings_response.text}\n")
        
        # Тест простого запроса к API
        print("Testing simple API request...")
        chat_response = requests.post(
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "GigaChat",
                "messages": [{"role": "user", "content": "Привет! Как дела?"}]
            },
            verify=False
        )
        print(f"Chat response status: {chat_response.status_code}")
        print(f"Chat response: {chat_response.text}\n")

if __name__ == "__main__":
    test_gigachat_api() 