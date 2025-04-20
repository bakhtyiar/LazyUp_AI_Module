import time
import tracemalloc

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from process_names.processes_log_loader import load_processes_logs

# Загрузка данных
data = load_processes_logs(1000)

# Преобразование в DataFrame
df = pd.DataFrame(data)
print(df.head())

mlb = MultiLabelBinarizer()
process_features = pd.DataFrame(mlb.fit_transform(df['processes']), columns=mlb.classes_)

# Объединение с исходными данными
df_processed = pd.concat([df.drop('processes', axis=1), process_features], axis=1)
print(df_processed.head())

# Удаление ненужных столбцов
df_processed = df_processed.drop(['process_categories', 'system_metrics', 'time_context'], axis=1)

print(df_processed.head())

X = df_processed.drop('mode', axis=1)  # Все признаки, кроме целевой переменной
y = df_processed['mode']  # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=100,  # Количество деревьев
    max_depth=10,  # Максимальная глубина дерева
    random_state=42
)

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()

model.fit(X_train, y_train)

end_train = time.time()
training_time = end_train - start_train

# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

# Оценка качества
sample_data = X_test
start_inf = time.time()

y_pred = model.predict(X_test)

end_inf = time.time()
inference_time = end_inf - start_inf
print(classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Inference time: {inference_time:.4f} s")
