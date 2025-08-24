import json
import re
from collections import defaultdict

def tokenize(text):
    """Улучшенная токенизация текста"""
    # Разбиваем на слова, сохраняем знаки препинания как отдельные токены
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def find_product_in_tokens(tokens, product):
    """Ищет товар в списке токенов (регистронезависимо)"""
    product_words = product.lower().split()
    product_len = len(product_words)
    
    matches = []
    for i in range(len(tokens) - product_len + 1):
        # Проверяем, совпадает ли последовательность токенов с товаром
        candidate = ' '.join(tokens[i:i+product_len]).lower()
        if candidate == product.lower():
            matches.append((i, i + product_len))
    
    return matches

def convert_to_iob(annotations_file, output_file):
    """Конвертирует размеченные данные в формат IOB2"""
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item_idx, item in enumerate(data):
            text = item['text']
            products = item['products']
            
            # Токенизируем текст
            tokens = tokenize(text)
            
            # Инициализируем все метки как 'O'
            labels = ['O'] * len(tokens)
            
            # Для каждого товара ищем его в токенах
            for product in products:
                matches = find_product_in_tokens(tokens, product)
                
                for start, end in matches:
                    # Размечаем начало как B-PRODUCT
                    labels[start] = 'B-PRODUCT'
                    # Размечаем продолжение как I-PRODUCT
                    for i in range(start + 1, end):
                        labels[i] = 'I-PRODUCT'
            
            # Записываем результат в формате IOB2
            for token, label in zip(tokens, labels):
                out_f.write(f"{token}\t{label}\n")
            out_f.write("\n")  # Пустая строка между предложениями
            
            # Выводим пример для первых 3 предложений
            if item_idx < 3:
                print(f"Пример {item_idx + 1}:")
                print(f"Текст: {text[:100]}...")
                print("Разметка (первые 20 токенов):")
                for i, (token, label) in enumerate(zip(tokens[:20], labels[:20])):
                    print(f"  {token} -> {label}")
                print()

# Конвертируем данные
convert_to_iob('quick_annotations.json', 'train_data.iob')
print("Данные преобразованы в формат IOB2 и сохранены в train_data.iob")

# Проверяем распределение меток
def check_label_distribution(iob_file):
    """Проверяет распределение меток в IOB файле"""
    label_counts = {'O': 0, 'B-PRODUCT': 0, 'I-PRODUCT': 0}
    
    with open(iob_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and '\t' in line:
                token, label = line.strip().split('\t')
                if label in label_counts:
                    label_counts[label] += 1
    
    total = sum(label_counts.values())
    print("Распределение меток:")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/total*100:.1f}%)")

check_label_distribution('train_data.iob')