import pandas as pd
import numpy as np

# Читаем собранные данные с обработкой возможных ошибок
try:
    # Параметр na_filter важен! Он говорит pandas не заменять пустые строки на NaN
    df = pd.read_csv('collected_texts.csv', na_filter=False)
    print(f"Успешно прочитано строк из CSV: {len(df)}")
except Exception as e:
    print(f"Ошибка при чтении CSV: {e}")
    exit()

# Создаем файл для удобной разметки
valid_count = 0
empty_count = 0

with open('for_annotation.txt', 'w', encoding='utf-8') as f:
    for i, row in df.iterrows():
        url = row['url']
        text = row['text']
        
        # Проверяем, что текст не пустой и не NaN
        if not text or str(text).strip().lower() in ['nan', 'null', 'none', '']:
            print(f"Пропускаем URL {i+1} (пустой текст): {url}")
            empty_count += 1
            continue
            
        # Проверяем минимальную длину текста
        if len(str(text).strip()) < 100:
            print(f"Пропускаем URL {i+1} (мало текста): {url}")
            empty_count += 1
            continue
            
        f.write(f"URL: {url}\n")
        f.write(f"TEXT {i+1}:\n{text}\n")
        f.write("="*80 + "\n\n")
        valid_count += 1

print(f"\nСтатистика:")
print(f"Всего строк в CSV: {len(df)}")
print(f"Валидных текстов для разметки: {valid_count}")
print(f"Пропущено (пустых/мало текста): {empty_count}")

# Создаем файл с проблемными URL для анализа
if empty_count > 0:
    with open('problematic_urls.txt', 'w', encoding='utf-8') as problem_file:
        problem_file.write("Пропущенные URL (пустой текст или мало контента):\n")
        for i, row in df.iterrows():
            text = row['text']
            if not text or str(text).strip().lower() in ['nan', 'null', 'none', ''] or len(str(text).strip()) < 100:
                problem_file.write(f"{row['url']}\n")
    print(f"Список проблемных URL сохранен в problematic_urls.txt")