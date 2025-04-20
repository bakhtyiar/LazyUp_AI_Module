import time
import tracemalloc

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from process_names.processes_log_loader import load_processes_logs

# Пример данных (можно заменить на загрузку из JSON)
data = load_processes_logs()

# Преобразуем в DataFrame
df = pd.DataFrame(data)


# Извлечем признаки из processes (частоты, уникальные и т. д.)
def extract_features(df):
    # Количество процессов
    df['process_count'] = df['processes'].apply(len)

    # Временные признаки из timestamp - добавляем обработку ошибок
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    # Если есть невалидные даты, заменяем их на текущую
    df['datetime'] = df['datetime'].fillna(pd.Timestamp.now())
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # One-Hot Encoding для процессов
    all_processes = set(process for sublist in df['processes'] for process in sublist)
    for process in all_processes:
        df[f'process_{process}'] = df['processes'].apply(lambda x: 1 if process in x else 0)

    return df.drop(columns=['processes', 'timestamp', 'datetime'])

df = extract_features(df)
# Удаление ненужных столбцов
df = df.drop(['process_categories', 'system_metrics', 'time_context'], axis=1)
X = df.drop(columns=['mode'])
y = df['mode']

# Разделим данные на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost требует числовых данных (у нас уже OHE)
model_xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()

model_xgb.fit(X_train, y_train)

end_train = time.time()
training_time = end_train - start_train
# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

# Оценка
start_inf = time.time()
y_pred_xgb = model_xgb.predict(X_test)
end_inf = time.time()
inference_time = end_inf - start_inf
print(classification_report(y_test, y_pred_xgb))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Inference time: {inference_time:.4f} s")

# def predict_mode(new_data):
#     # Пример нового объекта
#     new_entry = {
#         "processes": ["A", "X"],  # Новые процессы
#         "timestamp": 1620014400  # Новый timestamp
#     }
#
#     # Преобразуем в DataFrame и извлекаем признаки
#     new_df = pd.DataFrame([new_entry])
#     new_df = extract_features(new_df)
#     new_df = new_df[X_train.columns]  # Важно: сохранить порядок признаков
#
#     # Предсказание
#     mode_cb = model_cb.predict(new_df)[0]
#     mode_xgb = model_xgb.predict(new_df)[0]
#
#     return {"CatBoost": mode_cb, "XGBoost": mode_xgb}
#
#
# # Пример вызова
# print(predict_mode({"processes": ["A", "X"], "timestamp": 1620014400}))
#
