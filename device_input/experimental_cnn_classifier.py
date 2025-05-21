import time
import tracemalloc

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import optuna
import autokeras as ak

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


# 2. Создание модели CNN
def create_cnn_model(input_shape, num_classes=1, filters1=64, filters2=128, filters3=256, 
                    kernel_size=3, dense1_units=128, dense2_units=64, dropout1=0.5, dropout2=0.3):
    """
    Создает модель CNN для обработки временных последовательностей.

    Args:
        input_shape: Форма входных данных (max_sequence_length, n_features)
        num_classes: Количество классов (1 для бинарной классификации)
        filters1, filters2, filters3: Количество фильтров в сверточных слоях
        kernel_size: Размер ядра свертки
        dense1_units, dense2_units: Количество нейронов в полносвязных слоях
        dropout1, dropout2: Значения dropout

    Returns:
        Модель Keras
    """
    model = models.Sequential([
        # Нормализация входных данных
        layers.BatchNormalization(input_shape=input_shape),

        # Первый сверточный блок
        layers.Conv1D(filters1, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Второй сверточный блок
        layers.Conv1D(filters2, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Третий сверточный блок
        layers.Conv1D(filters3, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Полносвязные слои
        layers.Dense(dense1_units, activation='relu'),
        layers.Dropout(dropout1),
        layers.Dense(dense2_units, activation='relu'),
        layers.Dropout(dropout2),

        # Выходной слой
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model


# 3. Обучение модели с оптимизацией гиперпараметров Optuna
def train_with_optuna(X_train, y_train, X_val, y_val, input_shape, num_classes=1):
    """
    Обучает модель CNN с использованием оптимизации гиперпараметров от Optuna.

    Args:
        X_train: Обучающие данные
        y_train: Метки обучающих данных
        X_val: Валидационные данные
        y_val: Метки валидационных данных
        input_shape: Форма входных данных
        num_classes: Количество классов

    Returns:
        Лучшая модель, найденная с помощью Optuna
    """
    # Определяем функцию для оптимизации
    def objective(trial):
        # Определяем параметры для оптимизации
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'filters1': trial.suggest_categorical('filters1', [32, 64, 96, 128]),
            'filters2': trial.suggest_categorical('filters2', [64, 128, 192, 256]),
            'filters3': trial.suggest_categorical('filters3', [128, 256, 384, 512]),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'dense1_units': trial.suggest_categorical('dense1_units', [64, 128, 192, 256]),
            'dense2_units': trial.suggest_categorical('dense2_units', [32, 64, 96, 128]),
            'dropout1': trial.suggest_float('dropout1', 0.2, 0.7),
            'dropout2': trial.suggest_float('dropout2', 0.1, 0.5),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }
        
        # Создаем модель с текущими параметрами
        model = create_cnn_model(
            input_shape=input_shape,
            num_classes=num_classes,
            filters1=params['filters1'],
            filters2=params['filters2'],
            filters3=params['filters3'],
            kernel_size=params['kernel_size'],
            dense1_units=params['dense1_units'],
            dense2_units=params['dense2_units'],
            dropout1=params['dropout1'],
            dropout2=params['dropout2']
        )
        
        # Задаем learning rate для оптимизатора
        model.optimizer.learning_rate = params['learning_rate']
        
        # Добавляем callback для ранней остановки
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # Обучаем модель
        history = model.fit(
            X_train, y_train,
            epochs=50,  # Максимальное количество эпох
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0  # Отключаем вывод процесса обучения
        )
        
        # Возвращаем лучшее значение метрики на валидационных данных
        return max(history.history['val_accuracy'])
    
    # Создаем исследование Optuna
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    
    # Запускаем оптимизацию
    study.optimize(objective, n_trials=50)
    
    # Получаем лучшие параметры
    best_params = study.best_params
    print("Лучшие параметры, найденные Optuna:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Создаем и обучаем модель с лучшими параметрами
    best_model = create_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        filters1=best_params['filters1'],
        filters2=best_params['filters2'],
        filters3=best_params['filters3'],
        kernel_size=best_params['kernel_size'],
        dense1_units=best_params['dense1_units'],
        dense2_units=best_params['dense2_units'],
        dropout1=best_params['dropout1'],
        dropout2=best_params['dropout2']
    )
    
    best_model.optimizer.learning_rate = best_params['learning_rate']
    
    # Обучаем окончательную модель
    best_model.fit(
        X_train, y_train,
        epochs=100,  # Больше эпох для окончательной модели
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    return best_model


# 4. Neural Architecture Search с помощью AutoKeras
def neural_architecture_search(X_train, y_train, X_val, y_val, max_trials=10):
    """
    Выполняет поиск архитектуры нейронной сети с помощью AutoKeras.

    Args:
        X_train: Обучающие данные
        y_train: Метки обучающих данных
        X_val: Валидационные данные
        y_val: Метки валидационных данных
        max_trials: Максимальное количество итераций поиска

    Returns:
        Лучшая модель, найденная с помощью NAS
    """
    # Изменяем форму входных данных для AutoKeras
    # AutoKeras ожидает данные в формате (n_samples, n_timesteps, n_features)
    if len(X_train.shape) == 3:
        input_shape = X_train.shape[1:]
    else:
        raise ValueError("Неверная форма входных данных для AutoKeras")
    
    # Создаем модель AutoKeras для временных рядов
    ak_model = ak.StructuredDataClassifier(
        column_names=[f'feature_{i}' for i in range(X_train.shape[2])],
        loss='binary_crossentropy',
        objective='val_accuracy',
        max_trials=max_trials,  # Ограничиваем количество итераций поиска
        overwrite=True
    )
    
    # Преобразуем данные в нужный формат
    X_train_ak = X_train.reshape(X_train.shape[0], -1)
    X_val_ak = X_val.reshape(X_val.shape[0], -1)
    
    # Обучаем модель
    ak_model.fit(
        X_train_ak, y_train,
        validation_data=(X_val_ak, y_val),
        epochs=30
    )
    
    # Экспортируем лучшую модель
    best_model = ak_model.export_model()
    
    return best_model


# 5. Time Series Cross-Validation с помощью TimeSeriesSplit
def time_series_cross_validation(X, y, model_fn, n_splits=5):
    """
    Проводит кросс-валидацию временных рядов с помощью TimeSeriesSplit.

    Args:
        X: Входные данные
        y: Метки
        model_fn: Функция, создающая и обучающая модель
        n_splits: Количество разбиений для кросс-валидации

    Returns:
        Список метрик для каждого разбиения и среднее значение
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_index, test_index in tscv.split(X):
        # Разделяем данные на обучающие и тестовые для текущего разбиения
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Нормализуем данные
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Обучаем модель с помощью переданной функции
        model = model_fn(X_train, y_train, X_test, y_test)
        
        # Оцениваем модель
        loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
        scores.append({'loss': loss, 'accuracy': accuracy, 'auc': auc})
    
    # Рассчитываем средние метрики
    avg_scores = {
        'loss': np.mean([s['loss'] for s in scores]),
        'accuracy': np.mean([s['accuracy'] for s in scores]),
        'auc': np.mean([s['auc'] for s in scores])
    }
    
    return scores, avg_scores


# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    sample_data = load_device_logs(1000)
    
    # Предобработка данных
    X, y = preprocess_data(sample_data, max_sequence_length=50)
    
    # Разделение данных на обучающую и тестовую выборки
    # Используем последовательное разделение для временных рядов
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Разделение обучающей выборки на обучающую и валидационную
    val_size = int(0.2 * len(X_train))
    X_val, X_train = X_train[:val_size], X_train[val_size:]
    y_val, y_train = y_train[:val_size], y_train[val_size:]
    
    print("\n1. Обучение CNN с оптимизацией гиперпараметров Optuna")
    
    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()
    
    best_optuna_model = train_with_optuna(X_train, y_train, X_val, y_val, input_shape=X_train.shape[1:], num_classes=1)
    
    # Оценка модели на тестовых данных
    y_pred_optuna = (best_optuna_model.predict(X_test) > 0.5).astype("int32")
    print("\nРезультаты модели с оптимизацией Optuna:")
    print(classification_report(y_test, y_pred_optuna))
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Использование памяти: текущее = {current / 10**6:.2f}MB; пиковое = {peak / 10**6:.2f}MB")
    tracemalloc.stop()
