import numpy as np
from typing import List, Dict
from device_log_loader import load_device_logs


def extract_features(data: List[Dict[str, int]]) -> Dict[str, float]:
    """
    Извлекает статистические признаки из входных данных для последующей классификации.

    Args:
        data: Входные данные в формате:
            [{'buttonKey': int, 'dateTime': int}, ...]

    Returns:
        Словарь с вычисленными признаками:
        - count: количество записей
        - mean_time_diff: среднее время между нажатиями (в секундах)
        - median_time_diff: медианное время между нажатиями
        - min_key: минимальное значение buttonKey
        - max_key: максимальное значение buttonKey
        - freq_low: доля нажатий с buttonKey < 5
    """
    events = data
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

class ThresholdClassifier:
    def __init__(self):
        self.thresholds = None

    def fit(self, X, y):
        """Подбор порогов на обучающих данных"""
        # Здесь должна быть реализация подбора порогов
        # Для примера используем фиксированные значения
        self.thresholds = {
            'min_count': 3,
            'mean_time_thresh': 1000,
            'median_time_thresh': 800,
            'freq_low_thresh': 0.7,
            'key_range_thresh': 10
        }
        return self

    def predict(self, X):
        """Предсказание для новых данных"""
        return classify_by_thresholds(extract_features(X))

# Пример использования
if __name__ == "__main__":
    # Пример входных данных
    test_data = load_device_logs(100)
    for item in test_data:
        # Извлекаем признаки
        features = extract_features(test_data)
        # Классифицируем
        predicted_mode = classify_by_thresholds(features)
        print(item['mode'], predicted_mode)