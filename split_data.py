import random

def split_data(input_file, train_file, test_file, test_ratio=0.2):
    """Разделяет данные на обучающую и тестовую выборки"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разделяем на предложения (каждое предложение разделено пустой строкой)
    sentences = content.split('\n\n')
    
    # Убираем пустые предложения
    sentences = [s for s in sentences if s.strip()]
    
    # Перемешиваем случайным образом
    random.shuffle(sentences)
    
    # Разделяем на train/test
    split_idx = int(len(sentences) * (1 - test_ratio))
    train_data = sentences[:split_idx]
    test_data = sentences[split_idx:]
    
    # Сохраняем обучающую выборку
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_data))
    
    # Сохраняем тестовую выборку
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_data))
    
    print(f"Данные разделены: {len(train_data)} train, {len(test_data)} test")

split_data('train_data.iob', 'train.iob', 'test.iob')