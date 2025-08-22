import json
import re
from collections import defaultdict

def tokenize(text):
    """Простая токенизация текста на слова и знаки препинания"""
    return re.findall(r'\w+|[^\w\s]', text)

def find_product_positions(text, products):
    """Находит позиции товаров в тексте"""
    product_positions = []
    text_lower = text.lower()
    
    for product in products:
        product_lower = product.lower()
        start = 0
        while True:
            # Ищем товар в тексте (без учета регистра)
            pos = text_lower.find(product_lower, start)
            if pos == -1:
                break
                
            # Сохраняем позицию начала и конца
            end = pos + len(product_lower)
            product_positions.append((pos, end, product))
            start = end
    
    return product_positions

def convert_to_iob(annotations_file, output_file):
    """Конвертирует размеченные данные в формат IOB2"""
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in data:
            text = item['text']
            products = item['products']
            
            # Токенизируем текст
            tokens = tokenize(text)
            
            # Создаем словарь для меток каждого токена
            labels = ['O'] * len(tokens)
            
            # Находим позиции всех товаров в тексте
            product_positions = find_product_positions(text, products)
            
            # Размечаем токены, которые попадают в найденные позиции
            for start, end, product in product_positions:
                # Находим, какие токены попадают в этот диапазон
                current_pos = 0
                token_start_indices = []
                
                for i, token in enumerate(tokens):
                    token_start = text.find(token, current_pos)
                    token_end = token_start + len(token)
                    current_pos = token_end
                    
                    # Если токен полностью внутри найденного товара
                    if token_start >= start and token_end <= end:
                        token_start_indices.append(i)
                
                # Размечаем найденные токены
                if token_start_indices:
                    labels[token_start_indices[0]] = 'B-PRODUCT'
                    for idx in token_start_indices[1:]:
                        labels[idx] = 'I-PRODUCT'
            
            # Записываем результат в формате IOB2
            for token, label in zip(tokens, labels):
                out_f.write(f"{token}\t{label}\n")
            out_f.write("\n")  # Пустая строка между предложениями

# Конвертируем данные
convert_to_iob('quick_annotations.json', 'train_data.iob')
print("Данные преобразованы в формат IOB2 и сохранены в train_data.iob")