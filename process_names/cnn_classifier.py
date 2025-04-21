import time
import tracemalloc
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from process_names.processes_log_loader import load_processes_logs

data = load_processes_logs(10000)

df = pd.DataFrame(data)
print(df.head())

# Объединяем процессы в одну строку для каждого наблюдения
df["processes_text"] = df["processes"].apply(lambda x: " ".join(x))

# Создаем TF-IDF векторзатор
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df["processes_text"])

# Добавляем timestamp как числовой признак
X_timestamp = df[["timestamp"]].values

# Объединяем фичи
from scipy.sparse import hstack
X = hstack([X_text, X_timestamp])
y = df["mode"].values

# Токенизация процессов
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["processes_text"])
sequences = tokenizer.texts_to_sequences(df["processes_text"])

# Паддинг для одинаковой длины
max_len = max(len(x) for x in sequences)
X_seq = pad_sequences(sequences, maxlen=max_len, padding="post")

# Добавляем timestamp (нормализуем)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_time = scaler.fit_transform(df[["timestamp"]])

# Объединяем
X_final = np.hstack([X_seq, X_time])
y_final = df["mode"].values

from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input, Concatenate
from tensorflow.keras.models import Model

# Вход для процессов
input_seq = Input(shape=(max_len,))
embedding = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32)(input_seq)
conv1 = Conv1D(64, 3, activation="relu")(embedding)
pool1 = GlobalMaxPooling1D()(conv1)

# Вход для timestamp
input_time = Input(shape=(1,))

# Объединяем
merged = Concatenate()([pool1, input_time])
dense1 = Dense(32, activation="relu")(merged)
output = Dense(1, activation="sigmoid")(dense1)

model = Model(inputs=[input_seq, input_time], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Разделение данных
X_train_seq, X_test_seq, X_train_time, X_test_time, y_train, y_test = train_test_split(
    X_seq, X_time, y_final, test_size=0.2, random_state=42
)

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()
# Обучение
model.fit(
    [X_train_seq, X_train_time],
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_test_seq, X_test_time], y_test)
)
end_train = time.time()
training_time = end_train - start_train
# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()
# Оценка качества
start_inf = time.time()
y_pred = model.predict([X_test_seq, X_test_time])
end_inf = time.time()
inference_time = end_inf - start_inf
y_pred = [round(num) for sublist in y_pred for num in sublist]
print(classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Inference time: {inference_time:.4f} s")