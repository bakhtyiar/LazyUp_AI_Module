import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
import json
import os
import process_name_tokenizing.process_name_tokens_manager as pn_token_manager

def predict_by_process_names(df: pd.DataFrame, sequence_max_len=64, tokenizer_file=pn_token_manager.tokens_dict_filename):
    max_length = sequence_max_len  # Максимальная длина последовательности
    # Загрузить токенайзер и обновить его
    tokenizer = pn_token_manager.process_tokenization(df['processes'])
    # Выносим данные для предсказания
    X = tokenizer.texts_to_sequences(df['processes'])
    X = pad_sequences(X, maxlen=max_length)
    # y = pd.get_dummies(df['is_working_mode']).values  # one-hot encoding
    # Модель
    model = load_model('./predict_processes.h5')
    return model.predict(X)

def load_dataframe_process_names():
    log_data = []
    log_directory = './processes_logs'

    # Загрузка данных
    for filename in os.listdir(log_directory):
        if filename.endswith('.json'):  # Проверяем, что файл имеет расширение .json
            file_path = os.path.join(log_directory, filename)  # Полный путь к файлу
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)  # Загружаем данные из JSON-файла
                    # Проверяем структуру JSON
                    if (
                        isinstance(data, dict) and 
                        'is_working_mode' in data and 
                        'timestamp' in data and 
                        'processes' in data and 
                        isinstance(data['processes'], list)
                    ):
                        log_data.append(data)  # Добавляем данные в список
            except (json.JSONDecodeError, IOError) as e:
                print(f"Ошибка при обработке файла {filename}: {e}")
    data = log_data
    df = pd.DataFrame(data)
    return df

def main():
    df = load_dataframe_process_names()
    predict_by_process_names(df)

if __name__ == "__main__":
    main()