import time
import tracemalloc
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from process_names.processes_log_loader import load_processes_logs

# Загрузка данных
data = load_processes_logs(1000)
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

# Разделяем процессы и timestamp
X_train_process = X_train[:, :-1]
X_train_time = X_train[:, -1].reshape(-1, 1)
X_test_process = X_test[:, :-1]
X_test_time = X_test[:, -1].reshape(-1, 1)

# Функция для создания и обучения модели с заданными гиперпараметрами
def create_model(trial):
    # Параметры Embedding слоя
    embedding_dim = trial.suggest_int('embedding_dim', 16, 128)
    
    # Параметры GRU слоя
    gru_units = trial.suggest_int('gru_units', 32, 256)
    gru_recurrent_dropout = trial.suggest_float('gru_recurrent_dropout', 0.0, 0.5)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    
    # Параметры Dropout
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    
    # Параметры Dense слоя
    dense_units = trial.suggest_int('dense_units', 8, 128)
    
    # Параметры обучения
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    
    # Создание модели
    model = Sequential()
    model.add(Embedding(input_dim=len(all_processes)+1, output_dim=embedding_dim, input_length=max_len))
    
    # GRU слой (с опциональной двунаправленностью)
    if bidirectional:
        model.add(Bidirectional(GRU(gru_units, recurrent_dropout=gru_recurrent_dropout)))
    else:
        model.add(GRU(gru_units, recurrent_dropout=gru_recurrent_dropout))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    # Выбор оптимизатора
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Обучение модели
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=trial.suggest_int('early_stopping_patience', 3, 10),
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_process, y_train,
        epochs=trial.suggest_int('epochs', 5, 30),
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Оценка на тестовых данных
    y_pred_proba = model.predict(X_test_process)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Используем F1-score как метрику для оптимизации
    f1 = f1_score(y_test, y_pred)
    
    return model, f1, y_pred

def objective(trial):
    # Создаем и обучаем модель
    model, f1, _ = create_model(trial)
    
    # Освобождаем память GPU
    tf.keras.backend.clear_session()
    
    return f1

# Запуск оптимизации Optuna
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)

# Получение лучших параметров
best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best F1 score: {study.best_value}")

# Создание и оценка модели с лучшими параметрами
tracemalloc.start()
start_train = time.time()

best_model, _, y_pred = create_model(optuna.trial.FixedTrial(best_params))

end_train = time.time()
training_time = end_train - start_train
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

# Оценка качества
start_inf = time.time()
y_pred_proba = best_model.predict(X_test_process)
end_inf = time.time()
inference_time = end_inf - start_inf
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Training time: {training_time:.4f} s")
print(f"Inference time: {inference_time:.4f} s")