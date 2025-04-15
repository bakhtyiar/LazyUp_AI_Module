import time
import numpy as np
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import psutil
import os
import tracemalloc
from sklearn.model_selection import train_test_split
from device_input.device_log_loader import load_device_logs
from device_input.statistic_and_threshold_deepseek import ThresholdClassifier

class ClassifierMetrics:
    def __init__(self):
        self.metrics = {
            'accuracy': 0.,
            'precision': 0.,
            'recall': 0.,
            'f1': 0.,
            'roc_auc': 0.,
            'training_time': 0.,
            'avg_inference_time': 0.,
            'max_ram_usage': 0.,
            'gpu_usage': None
        }

    def evaluate(self, y_true: List[int], y_pred: List[int], y_scores: List[float] = None):
        """Вычисление стандартных метрик качества"""
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred)
        self.metrics['recall'] = recall_score(y_true, y_pred)
        self.metrics['f1'] = f1_score(y_true, y_pred)

        if y_scores is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        else:
            # Для порогового классификатора создаем псевдо-вероятности
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred)

    def measure_time(self, start_time: float, end_time: float, metric_name: str):
        """Запись времени выполнения"""
        self.metrics[metric_name] = end_time - start_time

    def measure_memory(self):
        """Измерение использования RAM"""
        process = psutil.Process(os.getpid())
        self.metrics['max_ram_usage'] = process.memory_info().rss / (1024 ** 2)  # в MB

    def get_metrics(self) -> Dict[str, float]:
        """Возвращает все собранные метрики"""
        return self.metrics


def run_benchmark(classifier, X_train, y_train, X_test, y_test):
    """Запуск полного бенчмарка классификатора"""
    metrics = ClassifierMetrics()

    # Измерение использования памяти до обучения
    tracemalloc.start()

    # Обучение модели с замером времени
    start_train = time.time()
    classifier.fit(X_train, y_train)
    end_train = time.time()
    metrics.measure_time(start_train, end_train, 'training_time')

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    metrics.metrics['max_ram_usage'] = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Предсказание на тестовых данных с замером времени
    inference_times = []
    y_pred = []
    y_scores = []

    for item in X_test:
        start_inf = time.time()
        pred = classifier.predict(X_test[0])
        end_inf = time.time()
        inference_times.append(end_inf - start_inf)
        y_pred.append(pred)

    # Для моделей с вероятностями
    if hasattr(classifier, 'predict_proba'):
        y_scores.append(classifier.predict_proba([X_test])[0][1])
    else:
        y_scores.append(pred)

    metrics.metrics['avg_inference_time'] = np.mean(inference_times)

    # Вычисление метрик качества
    metrics.evaluate(y_test, y_pred, y_scores if hasattr(classifier, 'predict_proba') else None)

    return metrics.get_metrics()

def evaluate(classifier, dataset):
    X = [item["list"] for item in dataset]
    y = [item["mode"] for item in dataset]

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создаем и тестируем классификатор
    metrics = run_benchmark(classifier(), X_train, y_train, X_test, y_test)

    return metrics

# Пример использования
if __name__ == "__main__":
    data = load_device_logs(1000)
    metrics = evaluate(ThresholdClassifier, data)
    # Выводим результаты
    print("\nРезультаты оценки классификатора:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\nTraining Time: {metrics['training_time']:.4f} sec")
    print(f"Avg Inference Time: {metrics['avg_inference_time']:.6f} sec/sample")
    print(f"Max RAM Usage: {metrics['max_ram_usage']:.2f} MB")
    print(f"GPU Usage: {metrics['gpu_usage'] or 'Not measured'}")