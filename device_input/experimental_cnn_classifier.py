import time
import tracemalloc

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

from device_input.device_log_loader import load_device_logs


# 1. Предобработка данных
def preprocess_data(data, max_sequence_length=100):
    """
    Преобразует сырые данные в формат, пригодный для обучения CNN.

    Args:
        data: Список словарей в формате {'mode': int, 'list': [{'buttonKey': int, 'dateTime': str}]}
        max_sequence_length: Максимальная длина последовательности (дополняется нулями)

    Returns:
        X: Тензор формы (n_samples, max_sequence_length, n_features)
        y: Массив меток
    """
    X = []
    y = []

    for item in data:
        # Извлекаем метку
        y.append(item['mode'])

        sequence = item['list']
        features = []

        # Обрабатываем каждое событие в последовательности
        for i, event in enumerate(sequence):
            if i >= max_sequence_length:
                break

            # Извлекаем признаки из каждого события
            button_key = event['buttonKey']
            timestamp = pd.to_datetime(event['dateTime'])

            # Вычисляем временную разницу с предыдущим событием
            if i > 0:
                prev_time = pd.to_datetime(sequence[i - 1]['dateTime'])
                time_diff = (timestamp - prev_time).total_seconds()
            else:
                time_diff = 0.0

            features.append([button_key, time_diff])

        # Дополняем последовательность нулями, если она короче max_sequence_length
        while len(features) < max_sequence_length:
            features.append([0, 0.0])

        X.append(features)

    return np.array(X), np.array(y)


# 2. Создание модели CNN
def create_cnn_model(input_shape, num_classes=1):
    """
    Создает модель CNN для обработки временных последовательностей.

    Args:
        input_shape: Форма входных данных (max_sequence_length, n_features)
        num_classes: Количество классов (1 для бинарной классификации)

    Returns:
        Модель Keras
    """
    model = models.Sequential([
        # Нормализация входных данных
        layers.BatchNormalization(input_shape=input_shape),

        # Первый сверточный блок
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Второй сверточный блок
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Третий сверточный блок
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Полносвязные слои
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        # Выходной слой
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model


# 3. Обучение модели
def train_model(X, y, epochs=20, batch_size=32):
    """
    Обучает модель CNN на предоставленных данных.

    Args:
        X: Входные данные
        y: Метки
        epochs: Количество эпох обучения
        batch_size: Размер батча

    Returns:
        Обученная модель и история обучения
    """

    # Создание модели
    model = create_cnn_model(input_shape=X.shape[1:])

    # Обучение модели
    history = model.fit(
        X, y,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)
    # Предобработка данных
    X, y = preprocess_data(sample_data, max_sequence_length=50)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Нормализация данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()

    # Обучение модели
    model, history = train_model(X_train, y_train, epochs=10)

    end_train = time.time()
    training_time = end_train - start_train
    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    # Сохранение модели
    model.save("button_sequence_classifier.h5")

    # Оценка качества
    start_inf = time.time()
    y_pred = model.predict(X_test)  # вызывать predict для отдельных строк
    end_inf = time.time()
    inference_time = end_inf - start_inf
    y_pred = [round(num) for sublist in y_pred for num in sublist]
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
