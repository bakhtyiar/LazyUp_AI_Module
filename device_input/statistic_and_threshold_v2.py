import time
import tracemalloc
from typing import List, Dict, Union

import numpy as np
from sklearn.metrics import classification_report

from device_input.device_log_loader import load_device_logs


def extract_features(data: Dict[str, Union[int, List[Dict[str, int]]]]) -> Dict[str, float]:
    """
    Извлекает статистические признаки из входных данных для последующей классификации.

    Args:
        data: Входные данные в формате {'mode': int, 'list': [{'buttonKey': int, 'dateTime': int}, ...]}

    Returns:
        Словарь с вычисленными признаками:
        - count: количество записей
        - mean_time_diff: среднее время между нажатиями (в секундах)
        - median_time_diff: медианное время между нажатиями
        - min_key: минимальное значение buttonKey
        - max_key: максимальное значение buttonKey
        - freq_low: доля нажатий с buttonKey < 5
    """
    events = data['list']
    timestamps = sorted([e['dateTime'] for e in events])
    button_keys = [e['buttonKey'] for e in events]

    # Вычисление временных промежутков между событиями
    time_diffs = np.diff(timestamps) if len(timestamps) > 1 else [0]

    features = {
        'count': len(events),
        'mean_time_diff': np.mean(time_diffs),
        'median_time_diff': np.median(time_diffs),
        'min_key': min(button_keys) if button_keys else 0,
        'max_key': max(button_keys) if button_keys else 0,
        'freq_low': sum(k < 5 for k in button_keys) / len(button_keys) if button_keys else 0
    }

    return features


def classify_by_thresholds(features: Dict[str, float]) -> int:
    """
    Классифицирует mode на основе извлеченных признаков и пороговых правил.

    Args:
        features: Словарь с извлеченными признаками

    Returns:
        Предсказанный mode (0 или 1)
    """
    # Эмпирически определенные пороговые значения
    if features['count'] < 3:
        return 0

    if (features['mean_time_diff'] > 1000 or
            features['median_time_diff'] > 800):
        return 1

    if features['freq_low'] > 0.7:
        return 0

    if features['max_key'] - features['min_key'] > 10:
        return 1

    # По умолчанию возвращаем 0
    return 0


# Пример использования
if __name__ == "__main__":
    # Пример входных данных
    sample_data = load_device_logs(1000)
    # Измерение использования памяти до
    tracemalloc.start()
    start_inf = time.time()
    # Предсказание на тестовых данных с замером времени
    y_pred = []
    for item in sample_data:
        # Извлекаем признаки
        features = extract_features(item)
        # Классифицируем
        y_pred.append(classify_by_thresholds(features))
    end_inf = time.time()
    inference_time = end_inf - start_inf
    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    y_test = [sample_data['mode'] for sample_data in sample_data]
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
