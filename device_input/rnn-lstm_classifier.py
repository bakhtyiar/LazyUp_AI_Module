import time
import tracemalloc

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
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


# Преобразование данных в numpy-массивы
def prepare_data(X, y):
    # Извлекаем buttonKey и dateTime из каждой последовательности
    X_processed = []
    for seq in X:
        seq_data = []
        for event in seq:
            seq_data.append([event['buttonKey'], event['dateTime']])
        X_processed.append(seq_data)

    # Приводим к numpy и нормализуем
    X_processed = np.array(X_processed, dtype='float32')
    scaler = MinMaxScaler()
    X_reshaped = X_processed.reshape(-1, 2)  # Объединяем все события для нормализации
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X_processed.shape)  # Возвращаем исходную структуру

    # Дополняем последовательности до одинаковой длины (если нужно)
    max_len = max(len(seq) for seq in X)
    X_padded = pad_sequences(X_scaled, maxlen=max_len, padding='post', dtype='float32')

    y = np.array(y, dtype='float32')
    return X_padded, y


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)
    X, y = prepare_dataset(sample_data)

    X_prepared, y_prepared = prepare_data(X, y)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_prepared, test_size=0.2, random_state=42)

    # Создание модели LSTM
    model = Sequential([
        Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])),  # Игнорирует нулевое дополнение
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Бинарная классификация
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()

    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32
    )
    end_train = time.time()
    training_time = end_train - start_train

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Оценка качества
    start_inf = time.time()
    y_pred = model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    y_pred = [round(num) for sublist in y_pred for num in sublist]
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
