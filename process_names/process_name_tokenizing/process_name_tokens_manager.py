import os
import json
from collections import defaultdict
from keras.preprocessing.text import Tokenizer

tokens_dict_filename = 'tokens_dictionary.json'

def load_or_create_tokenizer(filename='tokens_dictionary.json'):
    # Если файл существует, загружаем токенизатор
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            tokenizer_data = json.load(file)
            tokenizer = Tokenizer()
            tokenizer.word_index = tokenizer_data.get('word_index', {})
            tokenizer.index_word = {int(k): v for k, v in tokenizer_data.get('index_word', {}).items()}
            tokenizer.word_counts = defaultdict(int, tokenizer_data.get('word_counts', {}))
            tokenizer.word_docs = defaultdict(int, tokenizer_data.get('word_docs', {}))
            tokenizer.index_docs = defaultdict(int, {int(k): v for k, v in tokenizer_data.get('index_docs', {}).items()})
    else:
        # Если файл не существует, создаем новый токенизатор и сохраняем его в файл
        tokenizer = Tokenizer()
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump({
                'word_index': tokenizer.word_index,
                'index_word': tokenizer.index_word,
                'word_counts': dict(tokenizer.word_counts),
                'word_docs': dict(tokenizer.word_docs),
                'index_docs': dict(tokenizer.index_docs)
            }, file, ensure_ascii=False, indent=4)
    return tokenizer

def update_tokenizer(tokenizer: Tokenizer, texts: [str], filename='tokens_dictionary.json'):
    # Обновляем токенизатор новыми текстами
    tokenizer.fit_on_texts(texts)
    # Сохраняем обновленный токенизатор в файл
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump({
            'word_index': tokenizer.word_index,
            'index_word': tokenizer.index_word,
            'word_counts': dict(tokenizer.word_counts),
            'word_docs': dict(tokenizer.word_docs),
            'index_docs': dict(tokenizer.index_docs)
        }, file, ensure_ascii=False, indent=4)

def process_tokenization(texts: [str], filename='tokens_dictionary.json'):
    # Загружаем или создаем токенизатор
    tokenizer = load_or_create_tokenizer()
    # Обновляем токенизатор новыми текстами
    update_tokenizer(tokenizer, texts, filename)
    # Выводим текущий словарь токенов
    # print("Словарь токенов:", tokenizer.word_index)
    return tokenizer

def text_to_sequences(texts: [str], tokenizer: Tokenizer, filename='sequenced_texts.json'):
    print(tokenizer)
    ret = tokenizer.texts_to_sequences(texts)
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump({
            "seq": ret
        }, file, ensure_ascii=False, indent=4)
    return ret

# Пример применения:
# process_tokenization(["новый текстк ", "другой текстик", "текст"])
# process_tokenization(["новый текст для обработки ", "еще один другой текст"])
# text_to_sequences(["текст обработки"], load_or_create_tokenizer())
# ---