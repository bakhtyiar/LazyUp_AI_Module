import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime

from device_input.device_log_loader import load_device_logs


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


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)

    # Подготовка данных
    X, y = prepare_dataset(sample_data)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Для несбалансированных данных
    )
    model.fit(X_train, y_train)

    # Оценка качества
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # # Пример предсказания для новых данных
    # new_data = {
    #     "list": [
    #         {"buttonKey": 1, "dateTime": 1620002000},
    #         {"buttonKey": 1, "dateTime": 1620002003},
    #         {"buttonKey": 3, "dateTime": 1620002008}
    #     ]
    # }
    # new_features = extract_features(new_data['list'])
    # new_X = pd.DataFrame([new_features])
    # prediction = model.predict(new_X)
    # print(f"Predicted mode: {prediction[0]}")