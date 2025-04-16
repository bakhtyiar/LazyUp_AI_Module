import time
import tracemalloc
from datetime import datetime

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding, Masking
from tensorflow.keras.models import Sequential
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
        """
        Инициализация GRU классификатора

        Параметры:
        - max_sequence_length: максимальная длина последовательности событий
        - num_button_keys: количество уникальных значений buttonKey (для Embedding слоя)
        """
        self.max_sequence_length = max_sequence_length
        self.num_button_keys = num_button_keys
        self.model = self._build_model()

    def _build_model(self):
        """Построение архитектуры GRU модели"""
        model = Sequential([
            # Маскировка для обработки последовательностей переменной длины
            Masking(mask_value=0., input_shape=(self.max_sequence_length, 2)),

            # Embedding слой для buttonKey
            Embedding(input_dim=self.num_button_keys + 1,  # +1 для нулевого padding
                      output_dim=16,
                      input_length=self.max_sequence_length),

            # Двунаправленная GRU с dropout для регуляризации
            GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            GRU(32, dropout=0.2, recurrent_dropout=0.2),

            # Полносвязные слои
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _preprocess_data(self, X_raw, y_raw):
        """
        Предобработка сырых данных в формат, пригодный для обучения

        Параметры:
        - X_raw: список последовательностей событий
        - y_raw: список меток классов

        Возвращает:
        - X_processed: numpy array формы (n_samples, max_sequence_length, 2)
        - y_processed: numpy array меток классов
        """
        processed_sequences = []

        for sequence in X_raw:
            # Преобразование timestamp в относительное время (секунды от первого события)
            timestamps = [event['dateTime'] for event in sequence]
            if isinstance(timestamps[0], str):
                timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp() for ts in timestamps]

            time_deltas = [0] + [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

            # Нормализация временных интервалов
            max_delta = max(time_deltas) if max(time_deltas) > 0 else 1
            normalized_deltas = [delta / max_delta for delta in time_deltas]

            # Сочетание buttonKey и нормализованного временного интервала
            sequence_features = [
                [event['buttonKey'], normalized_deltas[i]]
                for i, event in enumerate(sequence)
            ]

            processed_sequences.append(sequence_features)

        # Добавление padding до максимальной длины последовательности
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

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32):
        """
        Обучение модели

        Параметры:
        - X_train: тренировочные последовательности
        - y_train: тренировочные метки
        - X_val, y_val: валидационные данные (опционально)
        - epochs: количество эпох
        - batch_size: размер батча
        """
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
        """Оценка модели на тестовых данных"""
        X_processed, y_processed = self._preprocess_data(X_test, y_test)

        loss, accuracy = self.model.evaluate(X_processed, y_processed, verbose=0)
        y_pred = (self.model.predict(X_processed) > 0.5).astype("int32")

        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_processed, y_pred))

        return y_pred

    def predict(self, X_new):
        """Предсказание для новых данных"""
        X_processed, _ = self._preprocess_data(X_new, [0] * len(X_new))
        return (self.model.predict(X_processed) > 0.5).astype("int32")


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)
    X, y = prepare_dataset(sample_data)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()

    # Инициализация и обучение модели
    classifier = GRUClassifier(max_sequence_length=50, num_button_keys=10)
    classifier.train(X_train, y_train, epochs=10)

    end_train = time.time()
    training_time = end_train - start_train

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()


    # Оценка качества
    start_inf = time.time()
    new_data = [[{"buttonKey": 1, "dateTime": "2023-01-03 08:00:00"}]]
    y_pred = classifier.predict(new_data)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
