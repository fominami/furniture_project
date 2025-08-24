
import pandas as pd

# Читаем собранные данные
df = pd.read_csv('collected_texts.csv')

# Сортируем тексты по длине (самые содержательные сначала)
df = df.dropna()  # Удаляем пустые
df['text_length'] = df['text'].apply(len)
df = df.sort_values('text_length', ascending=False)

# Выбираем топ-7 самых длинных текстов
top_texts = df.head(7)

# Сохраняем только лучшие тексты для разметки
with open('for_annotation_quick.txt', 'w', encoding='utf-8') as f:
    for i, row in top_texts.iterrows():
        f.write(f"URL: {row['url']}\n")
        f.write(f"TEXT {i+1}:\n{row['text']}\n")
        f.write("="*80 + "\n\n")

print(f"Отобрано {len(top_texts)} лучших текстов для быстрой разметки")