import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

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

    # Нормализованные частоты нажатий
    for button, count in button_counts.items():
        features[f'button_{button}_freq'] = count / features['count']

    # Временные признаки
    timestamps = [datetime.fromisoformat(d['dateTime']) for d in data]
    timestamps.sort()
    time_diffs = [(timestamps[i + 1] - timestamps[i]).total_seconds()
                  for i in range(len(timestamps) - 1)]

    if time_diffs:
        features.update({
            'time_mean': np.mean(time_diffs),
            'time_std': np.std(time_diffs),
            'time_max': max(time_diffs),
            'time_min': min(time_diffs),
            'time_median': np.median(time_diffs)
        })

    # Паттерны активности
    features['rapid_clicks'] = sum(1 for diff in time_diffs if diff < 2)  # Быстрые нажатия

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
def train_xgboost_model(X, y):
    """Обучает классификатор XGBoost"""

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

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

    # Оценка
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    return model


# Пример использования
if __name__ == "__main__":
    # Пример входных данных
    sample_data = load_device_logs(1000)

    # Подготовка данных
    X, y = prepare_dataset(sample_data)
    print("Извлеченные признаки:\n", X.head())

    # Обучение модели
    model = train_xgboost_model(X, y)