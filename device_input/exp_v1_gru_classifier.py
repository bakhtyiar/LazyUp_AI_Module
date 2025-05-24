import time
import tracemalloc
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (GRU, Dense, Dropout, Embedding,
                                     Masking, Lambda, Concatenate, Reshape, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from device_input.device_log_loader import load_device_logs


def prepare_dataset(json_data):
    """Подготавливает датасет из сырых JSON данных"""
    X = []
    y = []

    for item in json_data:
        # Извлекаем целевой признак
        y.append(item['mode'])

        # Извлекаем признаки из списка событий
        X.append(item['list'])

    return X, y


class GRUClassifier:
    def __init__(self, max_sequence_length=100, num_button_keys=100):
        self.max_sequence_length = max_sequence_length
        self.num_button_keys = num_button_keys
        self.model = self._build_model()

    def _build_model(self):
        """Построение архитектуры модели с исправленной обработкой типов данных"""
        # Входной слой
        input_layer = Input(shape=(self.max_sequence_length, 2))

        # Маскировка
        masked = Masking(mask_value=0.)(input_layer)

        # Разделение входных данных
        button_key = Lambda(lambda x: x[:, :, 0])(masked)
        time_delta = Lambda(lambda x: x[:, :, 1])(masked)

        # Преобразование buttonKey в целочисленный тип
        button_key_int = Lambda(lambda x: tf.cast(x, 'int32'))(button_key)

        # Embedding слой с явным указанием типа
        embedded = Embedding(
            input_dim=self.num_button_keys + 1,
            output_dim=16,
            input_length=self.max_sequence_length,
            mask_zero=True
        )(button_key_int)

        # Обработка временных меток
        time_processed = Reshape((self.max_sequence_length, 1))(time_delta)

        # Объединение признаков
        concatenated = Concatenate(axis=-1)([embedded, time_processed])

        # GRU слои
        gru1 = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(concatenated)
        gru2 = GRU(32, dropout=0.2, recurrent_dropout=0.2)(gru1)

        # Полносвязные слои
        dense1 = Dense(32, activation='relu')(gru2)
        dropout = Dropout(0.5)(dense1)
        output = Dense(1, activation='sigmoid')(dropout)

        model = Model(inputs=input_layer, outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _preprocess_data(self, X_raw, y_raw):
        """Исправленная предобработка данных с контролем типов"""
        processed_sequences = []

        for sequence in X_raw:
            # Обработка временных меток
            timestamps = [event['dateTime'] for event in sequence]
            if isinstance(timestamps[0], str):
                timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp() for ts in timestamps]

            time_deltas = [0] + [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
            max_delta = max(time_deltas) if max(time_deltas) > 0 else 1
            normalized_deltas = [delta / max_delta for delta in time_deltas]

            # Гарантируем целочисленный тип для buttonKey
            sequence_features = [
                [int(event['buttonKey']), float(normalized_deltas[i])]
                for i, event in enumerate(sequence)
            ]

            processed_sequences.append(sequence_features)

        # Паддинг с явным указанием типа float32
        X_padded = pad_sequences(
            processed_sequences,
            maxlen=self.max_sequence_length,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )

        y_array = np.array(y_raw, dtype='float32')

        return X_padded, y_array

    # Остальные методы класса остаются без изменений
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
        X_processed, y_processed = self._preprocess_data(X_train, y_train)

        if X_val is not None:
            X_val_processed, y_val_processed = self._preprocess_data(X_val, y_val)
            validation_data = (X_val_processed, y_val_processed)
        else:
            validation_data = None

        history = self.model.fit(
            X_processed, y_processed,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        X_processed, y_processed = self._preprocess_data(X_test, y_test)
        loss, accuracy = self.model.evaluate(X_processed, y_processed, verbose=0)
        y_pred = (self.model.predict(X_processed) > 0.5).astype("int32")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_processed, y_pred))
        return y_pred

    def predict(self, X_new):
        X_processed, _ = self._preprocess_data(X_new, [0] * len(X_new))
        return (self.model.predict(X_processed) > 0.5).astype("int32")


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)
    X, y = prepare_dataset(sample_data)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()

    # Инициализация и тестирование модели
    classifier = GRUClassifier(max_sequence_length=50, num_button_keys=80)
    classifier.train(X_train, y_train, epochs=5)
    end_train = time.time()
    training_time = end_train - start_train

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Оценка качества
    start_inf = time.time()
    y_pred = classifier.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    y_pred = [round(num) for sublist in y_pred for num in sublist]
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
