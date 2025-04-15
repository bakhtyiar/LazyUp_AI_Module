import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import time
import matplotlib.pyplot as plt
import sys

from device_input.device_log_loader import load_device_logs


# Функция извлечения признаков
def extract_features(data):
    features = []

    for item in data:
        list_events = item['list']
        num_events = len(list_events)
        button_counts = {}
        time_diffs = []

        # Извлечение временных меток и кнопок
        timestamps = [e['dateTime'] for e in list_events]
        buttons = [e['buttonKey'] for e in list_events]

        # Частотные признаки
        for btn in buttons:
            button_counts[btn] = button_counts.get(btn, 0) + 1

        # Временные признаки
        if len(timestamps) > 1:
            time_diff = [(timestamps[i - 1] - timestamps[i]).total_seconds()
                         for i in range(1, len(timestamps))]
            time_diffs = time_diff
        else:
            time_diffs = [0]

        # Формирование вектора признаков
        feature_vector = {
            'num_events': num_events,
            'time_mean': np.mean(time_diffs),
            'time_var': np.var(time_diffs),
            'btn5_ratio': button_counts.get(5, 0) / num_events,
            'btn1_count': button_counts.get(1, 0),
            'btn2_count': button_counts.get(2, 0),
            'time_pattern': 1 if any(np.array(time_diffs) < 5) else 0
        }

        features.append((feature_vector, item['mode']))

    return pd.DataFrame([f[0] for f in features]), pd.Series([f[1] for f in features])


# Измерение нагрузки на систему
def measure_performance(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # Метрики качества
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'inference_time_sec': inference_time,
        'model_size_mb': sys.getsizeof(model) / (1024 * 1024)
    }

    # Визуализация матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

    return metrics


class LogistricRegressionClassifier:
    def __init__(self, penalty='l2', C=0.1, solver='lbfgs', class_weight='balanced', max_iter=1000):
        self.classifier = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Стандартизация данных
        X_scaled = self.scaler.fit_transform(X)
        # Обучение модели
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X):
        # Стандартизация данных
        X_scaled = self.scaler.transform(X)
        # Предсказание
        return self.classifier.predict(X_scaled)


# Основной пайплайн
def main():
    # Генерация и преобразование данных
    raw_data = load_device_logs(1000)
    X, y = extract_features(raw_data)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Создание и обучение модели
    start_train = time.time()
    model = LogistricRegressionClassifier()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Оценка производительности
    metrics = measure_performance(model, X_test, y_test)
    metrics['train_time_sec'] = train_time

    # Вывод результатов
    print("Метрики классификации:")
    for k, v in metrics.items():
        print(f"{k:>20}: {v:.4f}")

    # Анализ важности признаков
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.classifier.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)

    print("\nВажность признаков:")
    print(feature_importance)

if __name__ == "__main__":
    main()