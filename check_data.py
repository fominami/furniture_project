# Создайте файл check_data.py и запустите его
import torch
import numpy as np
import random

# Фиксируем сиды
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
    print(f"Файл: {iob_file}")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/total*100:.1f}%)")
    return label_counts

print("=== ПРОВЕРКА ДАННЫХ ===")
train_stats = check_label_distribution("train.iob")
test_stats = check_label_distribution("test.iob")

# Проверим, есть ли PRODUCT в тестовых данных
if test_stats['B-PRODUCT'] + test_stats['I-PRODUCT'] == 0:
    print("❌ КРИТИЧЕСКАЯ ОШИБКА: В тестовых данных нет меток PRODUCT!")
else:
    print("✅ В тестовых данных есть метки PRODUCT")