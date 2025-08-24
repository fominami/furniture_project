
from flask import Flask, render_template, request, jsonify
from util_model import ner_model
from scraper_to_show import download_page, clean_html
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    """Главная страница с формой"""
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_products():
    """API endpoint для извлечения товаров"""
    url = request.form.get('url')
    
    if not url:
        return jsonify({'error': 'URL не предоставлен'}), 400
    
    try:
        # Шаг 1: Скачиваем страницу
        start_time = time.time()
        html_content = download_page(url)
        if not html_content:
            return jsonify({'error': 'Не удалось загрузить страницу'}), 500
        
        download_time = time.time() - start_time
        
        # Шаг 2: Очищаем HTML и извлекаем текст
        start_time = time.time()
        text = clean_html(html_content)
        if not text:
            return jsonify({'error': 'Не удалось извлечь текст из страницы'}), 500
        
        cleaning_time = time.time() - start_time
        
        # Шаг 3: Извлекаем товары с помощью модели
        start_time = time.time()
        if ner_model is None:
            return jsonify({'error': 'Модель не загружена'}), 500
        
        products = ner_model.extract_products(text)
        processing_time = time.time() - start_time
        
        # Логируем результаты
        logger.info(f"Обработан URL: {url}")
        logger.info(f"Найдено товаров: {len(products)}")
        logger.info(f"Время обработки: {download_time + cleaning_time + processing_time:.2f} сек.")
        
        # Формируем ответ
        response = {
            'success': True,
            'url': url,
            'products': products,
            'stats': {
                'download_time': round(download_time, 2),
                'cleaning_time': round(cleaning_time, 2),
                'processing_time': round(processing_time, 2),
                'total_time': round(download_time + cleaning_time + processing_time, 2),
                'products_count': len(products)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Ошибка при обработке URL {url}: {str(e)}")
        return jsonify({'error': f'Произошла ошибка: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)