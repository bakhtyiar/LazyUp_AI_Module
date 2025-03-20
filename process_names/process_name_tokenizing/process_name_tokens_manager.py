import os
import json
from collections import defaultdict
from keras.preprocessing.text import Tokenizer

def load_or_create_tokenizer(filename='tokenized_process_names.json'):
    # Если файл существует, загружаем токенизатор
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            tokenizer_data = json.load(file)
            tokenizer = Tokenizer()
            tokenizer.word_index = tokenizer_data.get('word_index', {})
            # tokenizer.index_word = {int(k): v for k, v in tokenizer_data.get('index_word', {}).items()}
            # tokenizer.word_counts = defaultdict(int, tokenizer_data.get('word_counts', {}))
            # tokenizer.word_docs = defaultdict(int, tokenizer_data.get('word_docs', {}))
            # tokenizer.index_docs = defaultdict(int, tokenizer_data.get('index_docs', {}))
    else:
        # Если файл не существует, создаем новый токенизатор и сохраняем его в файл
        tokenizer = Tokenizer()
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump({
                'word_index': tokenizer.word_index,
                # 'index_word': tokenizer.index_word,
                # 'word_counts': dict(tokenizer.word_counts),
                # 'word_docs': dict(tokenizer.word_docs),
                # 'index_docs': dict(tokenizer.index_docs)
            }, file, ensure_ascii=False, indent=4)
    return tokenizer

def update_tokenizer(tokenizer, texts, filename='tokenized_process_names.json'):
    # Обновляем токенизатор новыми текстами
    tokenizer.fit_on_texts(texts)
    # Сохраняем обновленный токенизатор в файл
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump({
            'word_index': tokenizer.word_index,
            # 'index_word': tokenizer.index_word,
            # 'word_counts': dict(tokenizer.word_counts),
            # 'word_docs': dict(tokenizer.word_docs),
            # 'index_docs': dict(tokenizer.index_docs)
        }, file, ensure_ascii=False, indent=4)

def process_tokenization(texts, filename='tokenized_process_names.json'):
    # Загружаем или создаем токенизатор
    tokenizer = load_or_create_tokenizer()
    # Обновляем токенизатор новыми текстами
    update_tokenizer(tokenizer, texts, filename)
    # Выводим текущий словарь токенов
    print("Current tokenizer word index:", tokenizer.word_index)