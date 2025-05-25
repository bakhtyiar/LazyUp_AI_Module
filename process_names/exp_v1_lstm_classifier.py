import time
import tracemalloc
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf

from process_names.processes_log_loader import load_processes_logs

# Установка seed для воспроизводимости
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Загрузка данных
data = load_processes_logs(1000)
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
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=RANDOM_SEED)

# Разделяем процессы и timestamp
X_process_train = X_train[:, :-1]  # Все, кроме последнего столбца (timestamp)
X_time_train = X_train[:, -1]      # Последний столбец (timestamp)
X_process_test = X_test[:, :-1]
X_time_test = X_test[:, -1]

# Размерность входных данных
max_seq_length = X_process_train.shape[1]
vocab_size = len(label_encoder.classes_)


def create_model(trial):
    """Создание модели с параметрами, предложенными Optuna"""
    # Параметры для Embedding слоя
    embedding_dim = trial.suggest_int('embedding_dim', 16, 256)
    
    # Параметры для рекуррентного слоя
    rnn_type = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU', 'Bidirectional'])
    rnn_units = trial.suggest_int('rnn_units', 16, 256)
    rnn_layers = trial.suggest_int('rnn_layers', 1, 3)
    
    # Параметры для полносвязных слоев
    dense_layers = trial.suggest_int('dense_layers', 0, 2)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 8, 128) for i in range(dense_layers)]
    
    # Параметры регуляризации
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.5)
    
    # Параметры оптимизатора
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    # Создаем модель
    model = Sequential()
    
    # Добавляем Embedding слой
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, 
                       input_length=max_seq_length, mask_zero=True))
    
    # Добавляем рекуррентные слои
    for i in range(rnn_layers):
        return_sequences = i < rnn_layers - 1  # True для всех слоев, кроме последнего
        
        if rnn_type == 'LSTM':
            model.add(LSTM(rnn_units, return_sequences=return_sequences, 
                          dropout=dropout_rate, recurrent_dropout=recurrent_dropout))
        elif rnn_type == 'GRU':
            model.add(GRU(rnn_units, return_sequences=return_sequences, 
                         dropout=dropout_rate, recurrent_dropout=recurrent_dropout))
        elif rnn_type == 'Bidirectional':
            model.add(Bidirectional(LSTM(rnn_units, return_sequences=return_sequences, 
                                       dropout=dropout_rate, recurrent_dropout=recurrent_dropout)))
    
    # Добавляем полносвязные слои
    for units in dense_units:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Выходной слой
    model.add(Dense(1, activation='sigmoid'))
    
    # Настраиваем оптимизатор
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:  # sgd
        optimizer = SGD(learning_rate=learning_rate)
    
    # Компилируем модель
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def objective(trial):
    """Целевая функция для оптимизации Optuna"""
    # Параметры обучения
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    epochs = 20  # Фиксированное количество эпох, будем использовать early stopping
    
    # Создаем модель с параметрами из trial
    model = create_model(trial)
    
    # Настраиваем early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Используем кросс-валидацию для более надежной оценки
    n_folds = 3
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    
    for train_idx, val_idx in kf.split(X_process_train, y_train):
        X_train_fold, X_val_fold = X_process_train[train_idx], X_process_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Обучаем модель
        model.fit(
            X_train_fold,
            y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Оцениваем модель
        y_pred = (model.predict(X_val_fold) > 0.5).astype(int)
        score = f1_score(y_val_fold, y_pred, average='weighted')
        scores.append(score)
    
    # Возвращаем среднее значение f1-score по всем фолдам
    return np.mean(scores)


# Запускаем оптимизацию Optuna
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=20)  # Можно увеличить количество trials для лучших результатов

# Получаем лучшие параметры
best_params = study.best_params
best_score = study.best_value
print(f"Best F1 Score: {best_score:.4f}")
print("Best hyperparameters:")
for param, value in best_params.items():
    print(f"    {param}: {value}")

# Создаем и обучаем лучшую модель на всех тренировочных данных
best_model = create_model(study.best_trial)

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()

# Обучаем модель с лучшими параметрами
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
best_model.fit(
    X_process_train,
    y_train,
    epochs=20,
    batch_size=best_params['batch_size'],
    validation_data=(X_process_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

end_train = time.time()
training_time = end_train - start_train

# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

# Оценка качества
start_inf = time.time()
y_pred = (best_model.predict(X_process_test) > 0.5).astype(int)
end_inf = time.time()
inference_time = end_inf - start_inf

# Вывод результатов
print("\nРезультаты на тестовой выборке:")
print(classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Training time: {training_time:.4f} s")
print(f"Inference time: {inference_time:.4f} s")

# Сохраняем лучшую модель
best_model.save('process_names/best_lstm_model.h5')