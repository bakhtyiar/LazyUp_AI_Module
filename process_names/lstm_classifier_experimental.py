import time
import tracemalloc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

from process_names.processes_log_loader import load_processes_logs

# Пример данных
data = load_processes_logs(1000)

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Кодируем процессы в числовые индексы
processes = df["processes"].tolist()
unique_processes = set(p for seq in processes for p in seq)
label_encoder = LabelEncoder()
label_encoder.fit(list(unique_processes))

# Преобразуем последовательности процессов в числовой формат
X_sequences = [label_encoder.transform(seq) for seq in processes]
X_padded = pad_sequences(X_sequences, padding="post")  # Добавляем padding, чтобы все последовательности были одной длины

# Добавляем timestamp как дополнительный признак
X_timestamp = df["timestamp"].values.reshape(-1, 1)
scaler = StandardScaler()
X_timestamp_scaled = scaler.fit_transform(X_timestamp)

# Объединяем процессы и timestamp
X_final = np.hstack((X_padded, X_timestamp_scaled))

# Целевая переменная
y = df["mode"].values

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Разделяем процессы и timestamp
X_process_train = X_train[:, :-1]  # Все, кроме последнего столбца (timestamp)
X_time_train = X_train[:, -1]      # Последний столбец (timestamp)
X_process_test = X_test[:, :-1]
X_time_test = X_test[:, -1]

# Размерность входных данных
max_seq_length = X_process_train.shape[1]
vocab_size = len(label_encoder.classes_)

# Создаем LSTM модель
model = Sequential([
    Embedding(input_dim=vocab_size + 1, output_dim=64, input_length=max_seq_length),  # +1 для padding
    LSTM(64),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()
# Обучаем модель
model.fit(
    X_process_train,
    y_train,
    epochs=10,
    batch_size=2,
    validation_data=(X_process_test, y_test)
)
end_train = time.time()
training_time = end_train - start_train
# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()
# Оценка качества
start_inf = time.time()
X_process_test = X_test[:, :-1]  # Все столбцы, кроме последнего (процессы)
X_time_test = X_test[:, -1]      # Последний столбец (timestamp)
y_pred = model.predict(X_process_test)
end_inf = time.time()
inference_time = end_inf - start_inf
y_pred = [round(num) for sublist in y_pred for num in sublist]
print(classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Inference time: {inference_time:.4f} s")