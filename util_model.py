
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import numpy as np
from typing import List, Set

class ProductNER:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully")
    
    def extract_products(self, text):
        """Extract only specific furniture model names"""
        if not text:
            return []
        
        # Ограничить длину текста для обработки
        if len(text) > 8000:
            text = text[:8000]
        
        try:
            # Cначала расширенное извлечение на основе модели
            products = self.model_based_extraction(text)
            
            #Восстановить полные названия моделей из фрагментов
            reconstructed_products= self.reconstruct_model_names(products, text)
            
            #Строгая фильтрация только по полным названиям моделей
            final_products = self.filter_complete_models(reconstructed_products)
            
            #Если подходящие модели не найдены, извлечение на основе контекста.
            if not final_products:
                final_products = self.context_based_extraction(text)
            
            return final_products
            
        except Exception as e:
            print(f"Error in product extraction: {e}")
            return []
    
    def reconstruct_model_names(self, fragments: List[str], original_text: str) -> List[str]:
        """Reconstruct complete model names from fragments"""
        if not fragments:
            return []
        
        # Сортировать фрагменты по длине (сначала длиннее)
        fragments.sort(key=len, reverse=True)
        
        reconstructed = []
        used_fragments = set()
        
        #Первый проход: ищем очевидные полные имена
        for i, fragment in enumerate(fragments):
            if self.looks_like_complete_model(fragment):
                reconstructed.append(fragment)
                used_fragments.add(i)
        
        #Второй проход: попытка объединить фрагменты, которые встречаются вместе в тексте.
        for i, frag1 in enumerate(fragments):
            if i in used_fragments:
                continue
                
            for j, frag2 in enumerate(fragments):
                if j <= i or j in used_fragments:
                    continue
                
                #расположены ли фрагменты близко друг к другу в исходном тексте.
                if self.fragments_are_adjacent(frag1, frag2, original_text):
                    combined = f"{frag1} {frag2}".strip()
                    if self.looks_like_complete_model(combined):
                        reconstructed.append(combined)
                        used_fragments.update([i, j])
                        break
        
        #Добавление оставшихся фрагментов, похожих на модели.
        for i, fragment in enumerate(fragments):
            if i not in used_fragments and self.looks_like_complete_model(fragment):
                reconstructed.append(fragment)
        
        return list(set(reconstructed))
    
    def fragments_are_adjacent(self, frag1: str, frag2: str, text: str) -> bool:
        """Check if two fragments appear close together in text"""
        pos1 = text.find(frag1)
        pos2 = text.find(frag2)
        
        if pos1 == -1 or pos2 == -1:
            return False
        
        
        distance = abs(pos1 - pos2)
        return distance <= 100  
    
    def looks_like_complete_model(self, text: str) -> bool:
        """Check if text looks like a complete furniture model name"""
        if not text or len(text) < 10:  
            return False
        
        words = text.split()
        if len(words) < 2:  
            return False
        
        # Распространенные закономерности в полных названиях моделей
        model_indicators = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]', 
            r'\b\d+[cm|mm|in|ft]', 
            r'\b[Kk]ing\b|\b[Qq]ueen\b|\b[Tt]win\b|\b[Ff]ull\b',  
            r'\b[Ee]uro\b|\b[Tt]op\b|\b[Pp]lush\b|\b[Pp]remium\b', 
            r'\b[Ss]eries\b|\b[Mm]odel\b|\b[Tt]ype\b',  
            r'\b[Ll]ED\b|\b[Ll]ight\b|\b[Rr]eading\b',  
        ]
        
       # Проверка наличия модельных индикаторов
        has_model_pattern = any(re.search(pattern, text) for pattern in model_indicators)
        
       # Проверьте структуру бренда и модели
        has_brand_structure = (
            len(words) >= 2 and 
            words[0].istitle() and 
            words[1].istitle() and
            not any(word.lower() in ['the', 'and', 'for', 'with'] for word in words[:2])
        )
        
        return has_model_pattern or has_brand_structure
    
    def filter_complete_models(self, products: List[str]) -> List[str]:
        """Strict filtering for only complete model names"""
        filtered = []
        
        for product in products:
            if self.is_complete_furniture_model(product):
               # Очистите название продукта
                cleaned = self.clean_model_name(product)
                filtered.append(cleaned)
        
        return list(set(filtered))
    
    def is_complete_furniture_model(self, text: str) -> bool:
        """Strict check for complete furniture models"""
      
        words = text.split()
        if len(words) < 2:
            return False
        
        # Должны содержать конкретные модельные показатели
        model_keywords = [
            'led', 'light', 'reading', 'bed', 'mattress', 'table', 'chair',
            'sofa', 'desk', 'cabinet', 'stand', 'frame', 'mirror', 'dresser'
        ]
        
        # содержит ли текст ключевые слова, связанные с мебелью.
        lower_text = text.lower()
        has_furniture_keyword = any(keyword in lower_text for keyword in model_keywords)
        
        # Проверьте правильность структуры модели
        has_proper_structure = (
            any(word.istitle() for word in words) and  # At least one capitalized word
            not text.endswith('.') and  # Not a sentence fragment
            not any(word.lower() in ['the', 'and', 'for', 'with', 'this'] for word in words[:2])  # Not starting with articles
        )
        
        return has_furniture_keyword and has_proper_structure
    
    def clean_model_name(self, text: str) -> str:
        """Clean up model names by removing common prefixes/suffixes"""
        # Удалить общие вводные фразы
        prefixes = [
            'the ', 'this ', 'our ', 'a ', 'an ', 'featuring ', 'including ',
            'with ', 'without ', 'and ', 'or ', 'for '
        ]
        
        cleaned = text.strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Удалить конечные знаки препинания
        cleaned = re.sub(r'[.,;:]$', '', cleaned)
        
        return cleaned
    
    def context_based_extraction(self, text: str) -> List[str]:
        """Extract model names based on contextual patterns"""
        products = []
        
        # Шаблоны для полных названий моделей
        model_patterns = [
            r'\b[A-Z][a-zA-Z\s]{10,50}\b',  # Longer capitalized phrases
            r'\b[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\b',  
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b', 
            r'\b[\w\s]+\s+-\s+[\w\s]+\b',  # Patterns with dashes (e.g., "Model - Size")
        ]
        
        #Обратите внимание на ценовой контекст (модели часто находятся рядом с ценами)
        price_pattern = r'\$\d+|\d+\s*USD|\d+\s*€|\d+\s*GBP'
        
        try:
            # Извлечение с использованием шаблонов моделей
            for pattern in model_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if self.is_complete_furniture_model(match):
                        products.append(match)
            
            # Посмотрите цены на названия моделей
            for match in re.finditer(price_pattern, text):
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Извлечь потенциальные названия моделей из контекста
                lines = context.split('\n')
                for line in lines:
                    words = line.split()
                    if len(words) >= 3:
                        #  закономерности, похожие на модели
                        if any(word.istitle() for word in words) and any(
                            kw in line.lower() for kw in ['led', 'light', 'reading', 'bed']
                        ):
                            products.append(line.strip())
            
        except Exception as e:
            print(f"Context extraction error: {e}")
        
        return list(set(products))
    
    
    def model_based_extraction(self, text):
        """Advanced extraction using the trained model"""
        products = []
        
        try:
            # Разделить текст на управляемые фрагменты
            chunks = self.split_text_into_chunks(text, max_tokens=300)
            
            for chunk in chunks:
                chunk_products = self.process_chunk_with_model(chunk)
                products.extend(chunk_products)
                
        except Exception as e:
            print(f"Model-based extraction failed: {e}")
        
        return products
    
    def process_chunk_with_model(self, text):
        """Process a text chunk with the NER model"""
        try:
           # Токенизация с отображением смещения
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_offsets_mapping=True
            )
            
           # Получить смещение и удалить из входов
            offset_mapping = inputs['offset_mapping'].cpu().numpy()[0]
            inputs = {k: v for k, v in inputs.items() if k != 'offset_mapping'}
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Получение прогнозов
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            
            # Преобразовать прогнозы в метки
            predicted_labels = [self.model.config.id2label[p] for p in predictions]
            
            # Извлечение объектов
            products = []
            current_entity = ""
            start_pos = None
            
            for i, (label, offset) in enumerate(zip(predicted_labels, offset_mapping)):
                
                if offset[0] == 0 and offset[1] == 0:
                    continue
                
                if label == "B-PRODUCT":
                   # Сохранить предыдущий объект
                    if current_entity and start_pos is not None:
                        entity_text = text[start_pos:offset[0]].strip()
                        if entity_text:
                            products.append(entity_text)
                    
                  # Начать новую сущность
                    current_entity = ""
                    start_pos = offset[0]
                
                elif label == "I-PRODUCT" and start_pos is not None:
                    # Продолжить текущий объект
                    pass
                
                else:
                   # Сохранить текущую сущность, если таковая имеется
                    if current_entity and start_pos is not None:
                        entity_text = text[start_pos:offset[0]].strip()
                        if entity_text:
                            products.append(entity_text)
                    
                   # Перезагрузить
                    current_entity = ""
                    start_pos = None
            
            # Добавить последний объект, если существует
            if current_entity and start_pos is not None:
                entity_text = text[start_pos:offset_mapping[-1][1]].strip()
                if entity_text:
                    products.append(entity_text)
            
            return products
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return []
    
    def split_text_into_chunks(self, text, max_tokens=300):
        """Split text into chunks based on sentences"""
        try:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_tokens = len(sentence.split())
                
                if current_length + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks
            
        except Exception as e:
            print(f"Text splitting error: {e}")
            return [text]


try:
    ner_model = ProductNER("./ner_model_small")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    ner_model = None