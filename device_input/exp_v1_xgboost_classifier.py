import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from device_input.device_log_loader import load_device_logs


# Функция для извлечения признаков из сырых данных
def extract_features(data):
    """Преобразует сырые данные в DataFrame с признаками"""
    features = {
        'count': len(data),  # Общее количество нажатий
        'unique_buttons': len(set(d['buttonKey'] for d in data)),  # Уникальные кнопки
    }

    # Частотные признаки для кнопок
    button_counts = {}
    for d in data:
        button = d['buttonKey']
        button_counts[button] = button_counts.get(button, 0) + 1

    # Временные характеристики
    timestamps = [d['dateTime'] for d in data]  # Use timestamps directly
    timestamps.sort()

    if len(timestamps) > 1:
        time_diffs = [timestamps[i + 1] - timestamps[i]
                      for i in range(len(timestamps) - 1)]

        features.update({
            'time_mean': np.mean(time_diffs),
            'time_std': np.std(time_diffs),
            'time_max': max(time_diffs),
            'time_min': min(time_diffs),
            'time_median': np.median(time_diffs)
        })
        features['rapid_clicks'] = sum(1 for diff in time_diffs if diff < 2)
    else:
        # Default values when not enough timestamps
        features.update({
            'time_mean': 0,
            'time_std': 0,
            'time_max': 0,
            'time_min': 0,
            'time_median': 0,
            'rapid_clicks': 0
        })

    return pd.DataFrame([features])


# Подготовка датасета
def prepare_dataset(json_data):
    """Преобразует массив JSON объектов в обучающий датасет"""

    X = pd.DataFrame()
    y = []

    for item in json_data:
        features = extract_features(item['list'])
        X = pd.concat([X, features], ignore_index=True)
        y.append(item['mode'])

    # Заполнение пропусков (если временные признаки отсутствуют)
    X = X.fillna(0)

    return X, np.array(y)


# Обучение модели
def train_xgboost_model(X_train, y_train):
    """Обучает классификатор XGBoost"""
    # Инициализация модели
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    # Обучение
    model.fit(X_train, y_train)
    return model


# Пример использования
if __name__ == "__main__":
    # Пример входных данных
    sample_data = load_device_logs(1000)

    # Подготовка данных
    X, y = prepare_dataset(sample_data)
    print("Извлеченные признаки:\n", X.head())

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Измерение использования памяти до
    tracemalloc.start()
    start_train = time.time()

    # Обучение модели
    model = train_xgboost_model(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Оценка
    start_inf = time.time()
    y_pred = model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
