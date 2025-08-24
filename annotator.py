import re

def quick_annotate(text):
    """
    Быстрая разметка - только очевидные товары
    """
    print("=" * 60)
    print("ТЕКСТ ДЛЯ РАЗМЕТКИ:")
    # Показываем только начало текста
    preview = text[:300] + "..." if len(text) > 300 else text
    print(preview)
    print("=" * 60)
    
    # Быстрый ввод - только названия товаров
    products = input("Введите ЛИШЬ ОЧЕВИДНЫЕ названия товаров через точку с запятой (;): ")
    
    if not products.strip() or products.strip().lower() == 'нет':
        return []
        
    return [p.strip() for p in products.split(';') if p.strip()]

# Обработка файла
def fast_annotation():
    with open('for_annotation.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    texts = content.split('=' * 80)
    results = []
    
    for i, text_block in enumerate(texts):
        if not text_block.strip() or i >= 22:  # только первые 7
            continue
            
        url_match = re.search(r'URL: (.*?)\n', text_block)
        text_match = re.search(r'TEXT \d+:\n(.*?)$', text_block, re.DOTALL)
        
        if url_match and text_match:
            print(f"\n📝 РАЗМЕТКА {i+1}/22")
            products = quick_annotate(text_match.group(1).strip())
            
            if products:
                results.append({
                    'text': text_match.group(1).strip(),
                    'products': products
                })
    
    # Сохраняем результат
    import json
    with open('quick_annotations.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Готово! Размечено {len(results)} текстов за ~{len(results)*5} минут")

if __name__ == "__main__":
    fast_annotation()
