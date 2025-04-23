import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from device_input.device_log_loader import load_device_logs

class ButtonPatternClassifier:
    def __init__(self):
        """Инициализация пайплайна с масштабированием и логистической регрессией"""
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        )

    def prepare_dataset(self, json_data):
        X = []
        y = []

        for item in json_data:
            X.append(item['list'])
            y.append(item['mode'])

        return X, y

    def _extract_features(self, dataset):
        """Извлечение признаков из сырых данных"""
        features = []
        for events in dataset:
            # Базовые статистики
            num_events = len(events)
            button_keys = [e['buttonKey'] for e in events]
            timestamps = [e['dateTime'] for e in events]

            # Временные характеристики
            time_diffs = []
            if len(timestamps) > 1:
                time_diffs = [
                    (timestamps[i] - timestamps[i + 1])
                    for i in range(len(timestamps) - 1)
                ]

                # Частотные характеристики кнопок
                button_counts = {i: 0 for i in range(1, 81)}
                for btn in button_keys:
                    if btn in button_counts:
                        button_counts[btn] += 1

                # Формирование вектора признаков
                feature_vec = {
                    'session_length': num_events,
                    'time_mean': np.mean(time_diffs) if time_diffs else 0,
                    'time_std': np.std(time_diffs) if time_diffs else 0,
                    **{f'btn{i}_ratio': button_counts[i] / num_events for i in range(1, 81)},
                    'rapid_clicks': sum(1 for diff in time_diffs if diff < 1.0)
                }
                features.append(feature_vec)

        return pd.DataFrame(features)

    def fit(self, X, y):
        """Обучение модели на размеченных данных"""
        X_features = self._extract_features(X)
        self.model.fit(X_features, y)
        return self

    def predict(self, X):
        """Предсказание классов для новых данных"""
        X_features = self._extract_features(X)
        return self.model.predict(X_features)

    def predict_proba(self, X):
        """Предсказание вероятностей классов"""
        X_features = self._extract_features(X)
        return self.model.predict_proba(X_features)

    def evaluate(self, X, y):
        """Оценка качества модели"""
        X_features = self._extract_features(X)
        y_pred = self.model.predict(X_features)
        return classification_report(y, y_pred, output_dict=True)

# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)
    classifier = ButtonPatternClassifier()
    X_data, y_labels = classifier.prepare_dataset(sample_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)
    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()
    classifier.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train
    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    # Оценка качества
    sample_data = X_test
    start_inf = time.time()
    y_pred = classifier.predict(sample_data)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")