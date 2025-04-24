import os
import time
import tracemalloc

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from process_names.processes_log_loader import load_processes_logs

module_dir = Path(__file__).resolve().parent
model_path = os.path.join(module_dir, 'predict_processes.joblib')
log_directory = os.path.join(module_dir, 'processes_logs')

def train_model(data, retrain=False):
    # Преобразуем в DataFrame если данные не в формате DataFrame
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data

    # Преобразуем процессы в строку (для CountVectorizer)
    df["processes_str"] = df["processes"].apply(lambda x: " ".join(x))

    # Разделяем на признаки (X) и целевую переменную (y)
    X = df[["timestamp", "processes_str"]]
    y = df["mode"]

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if retrain and os.path.exists(model_path):
        # Загружаем существующую модель для дообучения
        model = joblib.load(model_path)
    else:
        # Создаём новую модель
        preprocessor = ColumnTransformer(
            transformers=[
                ("processes", CountVectorizer(binary=True), "processes_str"),
                ("timestamp", StandardScaler(), ["timestamp"]),
            ]
        )
        model = make_pipeline(
            preprocessor,
            LogisticRegression(random_state=42, solver="liblinear", warm_start=True)
        )

    # Измерение использования памяти
    tracemalloc.start()
    start_train = time.time()

    # Обучаем модель
    model.fit(X_train, y_train)
    # Сохраняем модель
    joblib.dump(model, model_path)

    end_train = time.time()
    training_time = end_train - start_train
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Оценка модели
    start_inf = time.time()
    y_pred = model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "max_ram_usage": max_ram_usage,
        "training_time": training_time,
        "inference_time": inference_time
    }

    return model, metrics

if __name__ == "__main__":
    # Загрузка исходных данных
    initial_data = load_processes_logs(1000)
    
    # Обучение модели
    model, metrics = train_model(initial_data)
    
    # Вывод метрик
    print("Accuracy:", metrics["accuracy"])
    print("\nClassification Report:\n", metrics["classification_report"])
    print(f"Max RAM Usage: {metrics['max_ram_usage']:.2f} MB")
    print(f"Training time: {metrics['training_time']:.4f} s")
    print(f"Inference time: {metrics['inference_time']:.4f} s")
