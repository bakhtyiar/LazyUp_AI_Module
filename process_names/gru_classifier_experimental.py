import time
import tracemalloc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

from process_names.processes_log_loader import load_processes_logs

# Пример данных (можно заменить на загрузку из файла)
data = load_processes_logs(1000)

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Кодируем процессы в числовые индексы
all_processes = list(set(p for sublist in df["processes"] for p in sublist))
process_to_idx = {p: i+1 for i, p in enumerate(all_processes)}  # 0 будет для padding

# Преобразуем процессы в последовательности индексов
X_process = df["processes"].apply(lambda x: [process_to_idx[p] for p in x]).values
X_time = df["timestamp"].values.reshape(-1, 1)
y = df["mode"].values

# Padding для выравнивания длины последовательностей
max_len = max(len(x) for x in X_process)
X_process_padded = pad_sequences(X_process, maxlen=max_len, padding='post', truncating='post')

# Объединяем процессы и timestamp в один массив признаков
X = np.hstack([X_process_padded, X_time])

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding

# Разделяем процессы и timestamp
X_train_process = X_train[:, :-1]
X_train_time = X_train[:, -1].reshape(-1, 1)
X_test_process = X_test[:, :-1]
X_test_time = X_test[:, -1].reshape(-1, 1)

# Создаем модель GRU
model_gru = Sequential([
    Embedding(input_dim=len(all_processes)+1, output_dim=32, input_length=max_len),
    GRU(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()
# Обучение
model_gru.fit(
    X_train_process, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_process, y_test)
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
X_time_test = X_test[:, -1]
y_pred = model_gru.predict(X_process_test)
end_inf = time.time()
inference_time = end_inf - start_inf
y_pred = [round(num) for sublist in y_pred for num in sublist]
print(classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Inference time: {inference_time:.4f} s")