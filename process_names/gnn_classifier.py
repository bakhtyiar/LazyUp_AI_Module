import time
import tracemalloc

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from process_names.processes_log_loader import load_processes_logs

# Пример данных (можно заменить на загрузку из файла)
data = load_processes_logs(1000)

# Преобразование в DataFrame
df = pd.DataFrame(data)

# Предобработка:
# 1. processes -> строка (для BoW)
# 2. timestamp -> числовой признак
df["processes_str"] = df["processes"].apply(lambda x: " ".join(x))
df["timestamp"] = df["timestamp"].astype(float)

# Разделение на X и y
X = df[["processes_str", "timestamp"]]
y = df["mode"]

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline для обработки текста (processes) и числовых данных (timestamp)
preprocessor = ColumnTransformer(
    transformers=[
        ("text", CountVectorizer(), "processes_str"),  # Bag-of-Words для процессов
        ("num", StandardScaler(), ["timestamp"]),  # Нормализация времени
    ]
)

# Модели для тестирования
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Обучение и оценка
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train
    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    # Оценка качества
    start_inf = time.time()
    y_pred = pipeline.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
