import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import classification_report
import json
import os

def predict_by_process_names(df, sequence_max_len):
    max_length = sequence_max_len if sequence_max_len is not None else 64  # Максимальная длина последовательности

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['processes'])

    # Функция для фильтрации названий неизвестных процессов
    def filter_unknown_processes(processes):
        return [process for process in processes if process in tokenizer.word_index]

    # Применяем фильтрацию к колонке 'processes'
    df['processes'] = df['processes'].apply(filter_unknown_processes)

    X = tokenizer.texts_to_sequences(df['processes'])
    X = pad_sequences(X, maxlen=max_length)
    y = pd.get_dummies(df['is_working_mode']).values  # one-hot encoding

    # Модель
    model = None
    try:
        model = load_model('./predict_processes.h5')
    except OSError:
        print("Saved model not found")

    return model.predict(X)