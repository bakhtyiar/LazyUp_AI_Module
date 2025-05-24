import time
import tracemalloc

import numpy as np
import optuna
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from device_input.device_log_loader import load_device_logs


def prepare_dataset(json_data):
    """Подготавливает датасет из сырых JSON данных"""
    X = []
    y = []

    for item in json_data:
        # Извлекаем целевой признак
        y.append(item['mode'])

        # Извлекаем признаки из списка событий
        X.append(item['list'])

    return X, y


# Преобразование данных в numpy-массивы
def prepare_data(X, y):
    # Извлекаем buttonKey и dateTime из каждой последовательности
    X_processed = []
    for seq in X:
        seq_data = []
        for event in seq:
            seq_data.append([float(event['buttonKey']), float(event['dateTime'])])
        X_processed.append(seq_data)

    # Находим максимальную длину последовательности
    max_len = max(len(seq) for seq in X_processed)
    
    # Дополняем каждую последовательность до максимальной длины
    X_padded = np.zeros((len(X_processed), max_len, 2), dtype='float32')
    for i, seq in enumerate(X_processed):
        for j, features in enumerate(seq):
            X_padded[i, j] = features
    
    # Нормализуем данные
    scaler = MinMaxScaler()
    original_shape = X_padded.shape
    X_reshaped = X_padded.reshape(-1, 2)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(original_shape)

    y = np.array(y, dtype='float32')
    return X_scaled, y


def create_model(trial, input_shape):
    """Создает модель LSTM с параметрами, предложенными Optuna"""
    # Параметры архитектуры
    n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    lstm_units = [trial.suggest_int(f'lstm_units_{i}', 16, 256) for i in range(n_lstm_layers)]
    lstm_activation = trial.suggest_categorical('lstm_activation', ['tanh', 'relu'])
    dropout_rates = [trial.suggest_float(f'dropout_{i}', 0.0, 0.5) for i in range(n_lstm_layers)]
    
    # Параметры выходного слоя
    n_dense_layers = trial.suggest_int('n_dense_layers', 1, 2)
    dense_units = [trial.suggest_int(f'dense_units_{i}', 8, 128) for i in range(n_dense_layers-1)] + [1]
    dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'tanh'])
    final_activation = 'sigmoid'  # Для бинарной классификации
    
    # Параметры оптимизатора
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Создание модели
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=input_shape))
    
    # Добавление LSTM слоев
    for i in range(n_lstm_layers):
        return_sequences = (i < n_lstm_layers - 1)
        if bidirectional:
            lstm_layer = Bidirectional(LSTM(lstm_units[i], activation=lstm_activation, 
                                           return_sequences=return_sequences))
        else:
            lstm_layer = LSTM(lstm_units[i], activation=lstm_activation, 
                             return_sequences=return_sequences)
        model.add(lstm_layer)
        model.add(Dropout(dropout_rates[i]))
    
    # Добавление Dense слоев
    for i in range(n_dense_layers-1):
        model.add(Dense(dense_units[i], activation=dense_activation))
    
    # Выходной слой
    model.add(Dense(dense_units[-1], activation=final_activation))
    
    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def objective(trial, X_train, X_test, y_train, y_test):
    """Функция оптимизации для Optuna"""
    # Параметры обучения
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50)
    patience = trial.suggest_int('patience', 3, 10)
    
    # Создание модели
    model = create_model(trial, (X_train.shape[1], X_train.shape[2]))
    
    # Обучение с ранней остановкой
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Оценка на тестовых данных
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred_binary)
    
    return f1


# Пример использования
if __name__ == "__main__":
    # Загрузка и подготовка данных
    sample_data = load_device_logs(1000)
    X, y = prepare_dataset(sample_data)
    X_prepared, y_prepared = prepare_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_prepared, test_size=0.2, random_state=42)
    
    # Создание и запуск Optuna-исследования
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=50)
    
    # Вывод лучших параметров
    print("Best parameters:", study.best_params)
    print(f"Best f1 score: {study.best_value:.4f}")
    
    # Обучение лучшей модели
    best_trial = study.best_trial
    best_model = create_model(best_trial, (X_train.shape[1], X_train.shape[2]))
    
    # Измерение использования памяти и времени
    tracemalloc.start()
    start_train = time.time()
    
    # Обучение лучшей модели
    best_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=best_trial.params['epochs'],
        batch_size=best_trial.params['batch_size'],
        callbacks=[EarlyStopping(monitor='val_loss', patience=best_trial.params['patience'], 
                                 restore_best_weights=True)]
    )
    
    end_train = time.time()
    training_time = end_train - start_train
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    
    # Оценка качества
    start_inf = time.time()
    y_pred = best_model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred_binary))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Training time: {training_time:.2f} s")
    print(f"Inference time: {inference_time:.4f} s")
