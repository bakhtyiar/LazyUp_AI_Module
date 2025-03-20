from keras.preprocessing.text import Tokenizer
import json

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'w') as file:
        json.dump(tokenizer.word_index, file)

def load_tokenizer(file_path):
    with open(file_path, 'r') as file:
        word_index = json.load(file)
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    return tokenizer

tokens_dict_filename = "tokenized_process_names.json"