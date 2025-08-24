
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report, f1_score
import json
from sklearn.utils.class_weight import compute_class_weight

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка предобученной модели для русского языка
model_name = "DeepPavlov/bert-base-cased-conversational"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загрузка и подготовка данных
def load_and_prepare_data(train_file, test_file):
    """Загрузка и подготовка данных в формате IOB2"""
    
    def read_iob_file(file_path):
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Пустая строка - конец предложения
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, label = parts
                        current_sentence.append((token, label))
        
        return sentences

    # Читаем данные
    train_sentences = read_iob_file(train_file)
    test_sentences = read_iob_file(test_file)
    
    print(f"Загружено {len(train_sentences)} обучающих предложений")
    print(f"Загружено {len(test_sentences)} тестовых предложений")
    
    return train_sentences, test_sentences

# Преобразование в формат для Hugging Face
def convert_to_hf_format(sentences, tokenizer):
    """Преобразование данных в формат, понятный библиотеке transformers"""
    
    texts = []
    labels = []
    
    # Маппинг меток в числовой формат
    label_map = {"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2}
    
    for sentence in sentences:
        tokens = [item[0] for item in sentence]
        text_labels = [label_map[item[1]] for item in sentence]
        
        # Токенизируем с учетом подтокенов
        tokenized_input = tokenizer(
            tokens,
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=256  # Уменьшили максимальную длину
        )
        
        # Выравниваем метки для подтокенов
        word_ids = tokenized_input.word_ids()
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Игнорируем специальные токены
            elif word_idx != previous_word_idx:
                label_ids.append(text_labels[word_idx])
            else:
                label_ids.append(-100)  # Игнорируем подтокены
            previous_word_idx = word_idx
        
        texts.append(tokenized_input)
        labels.append(label_ids)
    
    return texts, labels

# Метрики для оценки
def compute_metrics(p):
    """Вычисление метрик качества"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Убираем игнорируемые индексы
    true_predictions = [
        [str(p) for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [str(l) for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Обратный маппинг числовых меток в текстовые
    label_map = {0: "O", 1: "B-PRODUCT", 2: "I-PRODUCT"}
    
    true_predictions_text = [
        [label_map[int(p)] for p in prediction]
        for prediction in true_predictions
    ]
    true_labels_text = [
        [label_map[int(l)] for l in label]
        for label in true_labels
    ]
    
    # Вычисляем метрики
    try:
        report = classification_report(true_labels_text, true_predictions_text, zero_division=0)
        f1 = f1_score(true_labels_text, true_predictions_text, zero_division=0)
    except:
        report = "Cannot compute metrics"
        f1 = 0
    
    return {
        "f1": f1,
        "report": report
    }

def main():
    print("🚀 Начинаем обучение модели NER для маленького датасета...")
    
    # 1. Загрузка данных
    print("📊 Загрузка данных...")
    train_sentences, test_sentences = load_and_prepare_data("train.iob", "test.iob")
    
    if not train_sentences or not test_sentences:
        print("❌ Ошибка: Не удалось загрузить данные для обучения")
        return
    
    # 2. Подготовка данных
    print("🔧 Подготовка данных...")
    train_texts, train_labels = convert_to_hf_format(train_sentences, tokenizer)
    test_texts, test_labels = convert_to_hf_format(test_sentences, tokenizer)
    
    # 3. Создание датасетов
    train_dataset = Dataset.from_dict({
        "input_ids": [x["input_ids"] for x in train_texts],
        "attention_mask": [x["attention_mask"] for x in train_texts],
        "labels": train_labels
    })
    
    test_dataset = Dataset.from_dict({
        "input_ids": [x["input_ids"] for x in test_texts],
        "attention_mask": [x["attention_mask"] for x in test_texts],
        "labels": test_labels
    })
    
    # 4. Вычисление весов классов для борьбы с дисбалансом
    all_labels = []
    for sentence in train_sentences:
        for token, label in sentence:
            label_map = {"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2}
            all_labels.append(label_map[label])
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Веса классов: {class_weights}")
    
    # 5. Загрузка модели
    print("🤖 Загрузка модели...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,  # O, B-PRODUCT, I-PRODUCT
        id2label={0: "O", 1: "B-PRODUCT", 2: "I-PRODUCT"},
        label2id={"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2}
    )
    model.to(device)
    
    # 6. Кастомная функция потерь с учетом весов классов
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Игнорируем num_items_in_batch и другие дополнительные параметры
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    # 7. Настройка обучения
    training_args = TrainingArguments(
        output_dir="./ner_model_small",
        learning_rate=1e-5, 
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=30,  
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=5,
        report_to="none",
        no_cuda=False if torch.cuda.is_available() else True
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # 8. Создание тренера
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # 9. Обучение
    print("🎯 Начинаем обучение...")
    train_result = trainer.train()
    
    # 10. Сохранение модели
    trainer.save_model()
    tokenizer.save_pretrained("./ner_model_small")
    
    # 11. Оценка качества
    print("📈 Оценка качества модели...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print(f"Final F1 Score: {eval_results['eval_f1']:.3f}")
    print("\nClassification Report:")
    print(eval_results['eval_report'])
    print("="*50)
    
    # 12. Сохранение метрик
    with open("training_results_small.json", "w", encoding="utf-8") as f:
        json.dump({
            "f1_score": eval_results["eval_f1"],
            "report": eval_results["eval_report"],
            "train_loss": train_result.training_loss
        }, f, indent=2, ensure_ascii=False)
    
    print("✅ Обучение завершено! Модель сохранена в папке 'ner_model_small'")

if __name__ == "__main__":
    main()