import os
import time
import tracemalloc

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from pathlib import Path
from device_input.device_log_loader import load_device_logs

module_dir = Path(__file__).resolve().parent

directory_path = os.path.join(module_dir, 'device_input_logs')  # Путь к директории с JSON-файлами
model_path = os.path.join(module_dir, 'predict_device_input.h5')  # Путь к модели


def extract_features(data):
    """Извлекает признаки из списка событий"""
    features = {}

    # Базовые признаки
    features['count'] = len(data)

    # Временные признаки
    timestamps = [x['dateTime'] for x in data]
    time_diffs = np.diff(timestamps)

    if len(time_diffs) > 0:
        features['time_diff_mean'] = np.mean(time_diffs)
        features['time_diff_std'] = np.std(time_diffs)
        features['time_diff_max'] = np.max(time_diffs)
        features['time_diff_min'] = np.min(time_diffs)
    else:
        features.update({
            'time_diff_mean': 0,
            'time_diff_std': 0,
            'time_diff_max': 0,
            'time_diff_min': 0
        })

    # Частотные признаки
    button_keys = [x['buttonKey'] for x in data]
    unique_keys, counts = np.unique(button_keys, return_counts=True)
    key_counts = dict(zip(unique_keys, counts))

    # Добавляем количество нажатий для каждой кнопки (до 10)
    for i in range(1, 11):
        features[f'key_{i}_count'] = key_counts.get(i, 0)

    # Относительные частоты
    total_presses = len(button_keys)
    for i in range(1, 11):
        features[f'key_{i}_ratio'] = features[f'key_{i}_count'] / total_presses if total_presses > 0 else 0

    # Временные паттерны
    if len(time_diffs) > 0:
        features['fast_events_5s'] = np.sum(np.array(time_diffs) <= 5)
        features['fast_events_10s'] = np.sum(np.array(time_diffs) <= 10)
    else:
        features.update({
            'fast_events_5s': 0,
            'fast_events_10s': 0
        })

    return features


def prepare_dataset(json_data):
    """Подготавливает датасет из сырых JSON данных"""
    X = []
    y = []

    for item in json_data:
        # Извлекаем целевой признак
        y.append(item['mode'])

        # Извлекаем признаки из списка событий
        features = extract_features(item['list'])
        X.append(features)

    # Преобразуем в DataFrame
    feature_df = pd.DataFrame(X)
    return feature_df, np.array(y)


def load_existing_model():
    """Загружает существующую модель, если она есть"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def train_model(data, existing_model=None):
    """Обучает или дообучает модель на новых данных"""
    # Подготовка данных
    X, y = prepare_dataset(data)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Использование существующей модели или создание новой
    if existing_model is not None:
        model = existing_model
        # Дообучение на новых данных
        model.fit(X_train, y_train)
    else:
        # Создание новой модели
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
    
    # Оценка качества
    y_pred = model.predict(X_test)
    print("Отчет о классификации:")
    print(classification_report(y_test, y_pred))
    
    return model


# Пример использования
if __name__ == "__main__":
    # Загрузка существующей модели
    existing_model = load_existing_model()
    
    # Загрузка новых данных
    new_data = load_device_logs(1000)
    
    # Измерение использования памяти
    tracemalloc.start()
    start_train = time.time()
    
    # Обучение или дообучение модели
    model = train_model(new_data, existing_model)
    
    end_train = time.time()
    training_time = end_train - start_train
    
    # Измерение памяти
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    
    print(f"Время обучения: {training_time:.4f} с")
    print(f"Максимальное использование RAM: {max_ram_usage:.2f} MB")
    
    # Сохранение обновленной модели
    joblib.dump(model, model_path)
