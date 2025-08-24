
import requests
from bs4 import BeautifulSoup
import re

def download_page(url):
    """Скачивает содержимое веб-страницы"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Ошибка при скачивании страницы: {e}")
        return None

def clean_html(html_content):
    """Очищает HTML и извлекает текст"""
    if not html_content:
        return ""
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Удаляем ненужные элементы
        for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            script.decompose()
        
        # Удаляем все теги, оставляя только текст
        text = soup.get_text()
        
        # Очищаем текст от лишних пробелов и переносов строк
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text
    except Exception as e:
        print(f"Ошибка при очистке HTML: {e}")
        return ""