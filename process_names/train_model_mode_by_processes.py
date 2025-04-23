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

log_directory = os.path.join(module_dir, 'processes_logs')

# Пример данных
data = load_processes_logs(1000)

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Преобразуем процессы в строку (для CountVectorizer)
df["processes_str"] = df["processes"].apply(lambda x: " ".join(x))

# Разделяем на признаки (X) и целевую переменную (y)
X = df[["timestamp", "processes_str"]]
y = df["mode"]

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём пайплайн для обработки данных:
# - processes_str → CountVectorizer (бинарные признаки)
# - timestamp → StandardScaler (нормализация)
preprocessor = ColumnTransformer(
    transformers=[
        ("processes", CountVectorizer(binary=True), "processes_str"),
        ("timestamp", StandardScaler(), ["timestamp"]),
    ]
)

# Объединяем предобработку и модель
model = make_pipeline(
    preprocessor,
    LogisticRegression(random_state=42, solver="liblinear")
)

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()

# Обучаем модель
model.fit(X_train, y_train)
joblib.dump(model, module_dir + '/predict_processes.joblib')

end_train = time.time()
training_time = end_train - start_train
# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

start_inf = time.time()
# Предсказываем на тестовых данных
y_pred = model.predict(X_test)
end_inf = time.time()
inference_time = end_inf - start_inf

# Оценка модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Inference time: {inference_time:.4f} s")
