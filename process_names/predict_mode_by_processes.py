import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import classification_report
import json
import os
import process_name_tokenizer

def predict_by_process_names(df, sequence_max_len=64, tokenizer_file=process_name_tokenizer.tokens_dict_filename):
    max_length = sequence_max_len  # Максимальная длина последовательности

    try:
        tokenizer = process_name_tokenizer.load_tokenizer(tokenizer_file)
    except FileNotFoundError:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['processes'])
        process_name_tokenizer.save_tokenizer(tokenizer, tokenizer_file)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['processes'])

    # Функция для фильтрации названий неизвестных процессов
    def filter_unknown_processes(processes):
        return [process for process in processes if process in tokenizer.word_index]

    # Применяем фильтрацию к колонке 'processes'
    df['processes'] = df['processes'].apply(filter_unknown_processes)

    X = tokenizer.texts_to_sequences(df['processes'])
    X = pad_sequences(X, maxlen=max_length)
    # y = pd.get_dummies(df['is_working_mode']).values  # one-hot encoding

    # Модель
    model = None
    try:
        model = load_model('./predict_processes.h5')
    except OSError:
        print("Saved model not found")

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