import time
import tracemalloc

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from device_input.device_log_loader import load_device_logs


# 1. Предобработка данных
def preprocess_data(data, max_sequence_length=100):
    """
    Преобразует сырые данные в формат, пригодный для обучения CNN.

    Args:
        data: Список словарей в формате {'mode': int, 'list': [{'buttonKey': int, 'dateTime': str}]}
        max_sequence_length: Максимальная длина последовательности (дополняется нулями)

    Returns:
        X: Тензор формы (n_samples, max_sequence_length, n_features)
        y: Массив меток
    """
    X = []
    y = []

    for item in data:
        # Извлекаем метку
        y.append(item['mode'])

        sequence = item['list']
        features = []

        # Обрабатываем каждое событие в последовательности
        for i, event in enumerate(sequence):
            if i >= max_sequence_length:
                break

            # Извлекаем признаки из каждого события
            button_key = event['buttonKey']
            timestamp = pd.to_datetime(event['dateTime'])

            # Вычисляем временную разницу с предыдущим событием
            if i > 0:
                prev_time = pd.to_datetime(sequence[i - 1]['dateTime'])
                time_diff = (timestamp - prev_time).total_seconds()
            else:
                time_diff = 0.0

            features.append([button_key, time_diff])

        # Дополняем последовательность нулями, если она короче max_sequence_length
        while len(features) < max_sequence_length:
            features.append([0, 0.0])

        X.append(features)

    return np.array(X), np.array(y)


# 2. Создание модели CNN с Optuna
def create_cnn_model(trial, input_shape, num_classes=1):
    """
    Создает модель CNN для обработки временных последовательностей с оптимизацией гиперпараметров через Optuna.

    Args:
        trial: Объект trial из Optuna для оптимизации гиперпараметров
        input_shape: Форма входных данных (max_sequence_length, n_features)
        num_classes: Количество классов (1 для бинарной классификации)

    Returns:
        Модель Keras с оптимизированными гиперпараметрами
    """
    # Определение гиперпараметров модели
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 4)
    
    # Выбор оптимизатора и его параметров
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    # Создание модели
    model = models.Sequential()
    
    # Нормализация входных данных
    model.add(layers.BatchNormalization(input_shape=input_shape))
    
    # Динамическое создание сверточных блоков
    for i in range(n_conv_layers):
        # Количество фильтров увеличивается с глубиной сети
        filters = trial.suggest_int(f'filters_layer_{i}', 32, 256, step=32)
        kernel_size = trial.suggest_int(f'kernel_size_layer_{i}', 2, 5)
        activation = trial.suggest_categorical(f'activation_layer_{i}', ['relu', 'elu', 'selu'])
        
        model.add(layers.Conv1D(filters, kernel_size=kernel_size, activation=activation, padding='same'))
        model.add(layers.BatchNormalization())
        
        # Добавляем пулинг (не для последнего слоя)
        if i < n_conv_layers - 1:
            pool_size = trial.suggest_int(f'pool_size_layer_{i}', 2, 3)
            model.add(layers.MaxPooling1D(pool_size=pool_size))
    
    # Глобальный пулинг
    pooling_type = trial.suggest_categorical('global_pooling', ['avg', 'max'])
    if pooling_type == 'avg':
        model.add(layers.GlobalAveragePooling1D())
    else:
        model.add(layers.GlobalMaxPooling1D())
    
    # Полносвязные слои
    n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
    for i in range(n_dense_layers):
        units = trial.suggest_int(f'dense_units_{i}', 32, 256, step=32)
        activation = trial.suggest_categorical(f'dense_activation_{i}', ['relu', 'elu', 'selu'])
        model.add(layers.Dense(units, activation=activation))
        
        # Добавляем Dropout
        dropout_rate = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        model.add(layers.Dropout(dropout_rate))
    
    # Выходной слой
    model.add(layers.Dense(num_classes, activation='sigmoid'))
    
    # Компиляция модели
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model


# 3. Функция для оптимизации гиперпараметров с Optuna
def objective(trial, X_train, y_train, X_val, y_val):
    """
    Целевая функция для оптимизации гиперпараметров с Optuna.
    
    Args:
        trial: Объект trial из Optuna
        X_train: Обучающие данные
        y_train: Метки обучающих данных
        X_val: Валидационные данные
        y_val: Метки валидационных данных
        
    Returns:
        Значение метрики для оптимизации (AUC) и trial
    """
    # Параметры обучения
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    epochs = trial.suggest_int('epochs', 10, 30)
    
    # Создание модели с оптимизированными гиперпараметрами
    model = create_cnn_model(trial, input_shape=X_train.shape[1:])
    
    # Ранняя остановка
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=trial.suggest_int('patience', 3, 10),
        mode='max',
        restore_best_weights=True
    )
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Оценка модели на валидационных данных
    _, _, val_auc = model.evaluate(X_val, y_val, verbose=0)
    
    return val_auc, trial


# 4. Обучение модели с оптимальными гиперпараметрами
def train_optimized_model(X_train, y_train, X_val, y_val, best_trial):
    """
    Обучает модель CNN с оптимальными гиперпараметрами.
    
    Args:
        X_train: Обучающие данные
        y_train: Метки обучающих данных
        X_val: Валидационные данные
        y_val: Метки валидационных данных
        best_trial: Лучший trial из Optuna
        
    Returns:
        Обученная модель и история обучения
    """
    # Создаем модель с оптимальными гиперпараметрами
    model = create_cnn_model(best_trial, input_shape=X_train.shape[1:])
    
    # Ранняя остановка
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=best_trial.params['patience'],
        mode='max',
        restore_best_weights=True
    )
    
    # Обучение модели
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=best_trial.params['epochs'],
        batch_size=best_trial.params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history


# Пример использования
if __name__ == "__main__":
    # Загрузка и предобработка данных
    sample_data = load_device_logs(1000)
    X, y = preprocess_data(sample_data, max_sequence_length=50)
    
    # Разделение данных с использованием временного разделения (TimeSeriesSplit)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Получаем последний сплит для финального разделения (имитация реальных условий)
    train_index, test_index = list(tscv.split(X))[-1]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Дополнительное разделение обучающей выборки на обучающую и валидационную
    # также с учетом временной структуры
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Создание и запуск исследования Optuna
    print("Начало оптимизации гиперпараметров с Optuna...")
    study = optuna.create_study(direction='maximize', study_name='cnn_optimization')
    
    # Функция-обертка для целевой функции, возвращающей только значение AUC
    def objective_wrapper(trial):
        val_auc, _ = objective(trial, X_train, y_train, X_val, y_val)
        return val_auc
    
    study.optimize(objective_wrapper, n_trials=20)
    
    # Вывод лучших гиперпараметров
    print("Лучшие гиперпараметры:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()
    
    # Обучение модели с оптимальными гиперпараметрами
    print("\nОбучение модели с оптимальными гиперпараметрами...")
    model, history = train_optimized_model(X_train, y_train, X_val, y_val, study.best_trial)
    
    end_train = time.time()
    training_time = end_train - start_train
    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()
    
    # Сохранение модели
    model.save("exp_v1_cnn_classifier.h5")
    
    # Оценка качества на тестовых данных
    print("\nОценка качества на тестовых данных:")
    start_inf = time.time()
    y_pred = model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    y_pred = [round(num) for sublist in y_pred for num in sublist]
    print(classification_report(y_test, y_pred))
    print(f"Лучшее значение AUC: {study.best_value:.4f}")
    print(f"Training time: {training_time:.2f} с")
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
