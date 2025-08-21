import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import re

def get_page_html(url):
    """
    Загружает HTML-страницу по указанному URL
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return None

def extract_clean_text(html):
    """
    Извлекает и очищает текст из HTML
    """
    if not html:
        return ""
    
    soup = BeautifulSoup(html, 'lxml')
    
    # Удаляем ненужные элементы
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.decompose()
    
    # Ищем основной контент
    main_content = soup.find('main') or soup.find('article') or soup.body
    
    # Извлекаем текст
    text = main_content.get_text(separator=' ', strip=True)
    
    # Очищаем текст
    lines = (line.strip() for line in text.splitlines())
    clean_text = ' '.join(line for line in lines if line)
    
    # Удаляем множественные пробелы
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text

def read_urls_from_csv(filename, limit=30):
    """
    Читает URL из CSV-файла с ограничением количества
    """
    try:
        # Читаем CSV-файл
        df = pd.read_csv(filename, header=None, names=['data'])
        
        # Извлекаем URL из данных
        urls = []
        for index, row in df.iterrows():
            # Пропускаем первую строку с "max(page)"
            if index == 0 and 'max(page)' in row['data']:
                continue
                
            # Ищем URL в строке
            parts = row['data'].split()
            for part in parts:
                if part.startswith('http'):
                    urls.append(part)
                    break
        
        return urls[:limit]
    except Exception as e:
        print(f"Ошибка при чтении CSV: {e}")
        return []

def main():
    # Читаем URL из CSV-файла (первые 30)
    urls = read_urls_from_csv('URL_list.csv', limit=30)
    
    if not urls:
        print("Не удалось прочитать URL из файла")
        return
    
    print(f"Найдено {len(urls)} URL для обработки")
    
    # Собираем текст с каждой страницы
    all_texts = []
    
    for i, url in enumerate(urls, 1):
        print(f"Обрабатывается URL {i}/{len(urls)}: {url}")
        
        html = get_page_html(url)
        if not html:
            continue
            
        text = extract_clean_text(html)
        all_texts.append({
            'url': url,
            'text': text
        })
        
        # Пауза между запросами
        time.sleep(2)
    
    # Сохраняем результаты
    output_file = 'collected_texts.csv'
    df = pd.DataFrame(all_texts)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Готово! Результаты сохранены в {output_file}")
    
    # Статистика
    total_text_length = sum(len(item['text']) for item in all_texts)
    print(f"Общий объем собранного текста: {total_text_length} символов")
    print(f"Средняя длина текста на странице: {total_text_length // len(all_texts)} символов")

if __name__ == "__main__":
    main()