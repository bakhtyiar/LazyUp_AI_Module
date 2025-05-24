import time
import tracemalloc
from datetime import datetime

import numpy as np
import tensorflow as tf
import optuna
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.layers import (GRU, Dense, Dropout, Embedding,
                                     Masking, Lambda, Concatenate, Reshape, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
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


class GRUClassifier:
    def __init__(self, max_sequence_length=100, num_button_keys=100):
        self.max_sequence_length = max_sequence_length
        self.num_button_keys = num_button_keys
        self.model = None
        self.best_params = None

    def _build_model(self, trial=None):
        """Построение архитектуры модели с оптимизацией гиперпараметров через Optuna"""
        # Если trial не передан, используем сохраненные лучшие параметры
        if trial is None:
            if self.best_params is None:
                raise ValueError("Model has not been optimized yet. Call optimize() first.")
            params = self.best_params
        else:
            params = {
                # Параметры Embedding слоя
                'embedding_dim': trial.suggest_categorical('embedding_dim', [8, 16, 32, 64]),
                
                # Параметры GRU слоев
                'gru1_units': trial.suggest_categorical('gru1_units', [32, 64, 128, 256]),
                'gru1_dropout': trial.suggest_float('gru1_dropout', 0.0, 0.5),
                'gru1_recurrent_dropout': trial.suggest_float('gru1_recurrent_dropout', 0.0, 0.5),
                'gru2_units': trial.suggest_categorical('gru2_units', [16, 32, 64, 128]),
                'gru2_dropout': trial.suggest_float('gru2_dropout', 0.0, 0.5),
                'gru2_recurrent_dropout': trial.suggest_float('gru2_recurrent_dropout', 0.0, 0.5),
                
                # Параметры Dense слоев
                'dense_units': trial.suggest_categorical('dense_units', [16, 32, 64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.7),
                
                # Параметры оптимизатора
                'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'rmsprop', 'sgd']),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                
                # Параметры обучения
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            }

        # Входной слой
        input_layer = Input(shape=(self.max_sequence_length, 2))

        # Маскировка
        masked = Masking(mask_value=0.)(input_layer)

        # Разделение входных данных
        button_key = Lambda(lambda x: x[:, :, 0])(masked)
        time_delta = Lambda(lambda x: x[:, :, 1])(masked)

        # Преобразование buttonKey в целочисленный тип
        button_key_int = Lambda(lambda x: tf.cast(x, 'int32'))(button_key)

        # Embedding слой с явным указанием типа
        embedded = Embedding(
            input_dim=self.num_button_keys + 1,
            output_dim=params['embedding_dim'],
            input_length=self.max_sequence_length,
            mask_zero=True
        )(button_key_int)

        # Обработка временных меток
        time_processed = Reshape((self.max_sequence_length, 1))(time_delta)

        # Объединение признаков
        concatenated = Concatenate(axis=-1)([embedded, time_processed])

        # GRU слои
        gru1 = GRU(
            params['gru1_units'], 
            return_sequences=True, 
            dropout=params['gru1_dropout'], 
            recurrent_dropout=params['gru1_recurrent_dropout']
        )(concatenated)
        
        gru2 = GRU(
            params['gru2_units'], 
            dropout=params['gru2_dropout'], 
            recurrent_dropout=params['gru2_recurrent_dropout']
        )(gru1)

        # Полносвязные слои
        dense1 = Dense(params['dense_units'], activation='relu')(gru2)
        dropout = Dropout(params['dropout_rate'])(dense1)
        output = Dense(1, activation='sigmoid')(dropout)

        model = Model(inputs=input_layer, outputs=output)

        # Выбор оптимизатора
        if params['optimizer_name'] == 'adam':
            optimizer = Adam(learning_rate=params['learning_rate'])
        elif params['optimizer_name'] == 'rmsprop':
            optimizer = RMSprop(learning_rate=params['learning_rate'])
        else:  # sgd
            optimizer = SGD(learning_rate=params['learning_rate'])

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _preprocess_data(self, X_raw, y_raw):
        """Исправленная предобработка данных с контролем типов"""
        processed_sequences = []

        for sequence in X_raw:
            # Обработка временных меток
            timestamps = [event['dateTime'] for event in sequence]
            if isinstance(timestamps[0], str):
                timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').timestamp() for ts in timestamps]

            time_deltas = [0] + [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
            max_delta = max(time_deltas) if max(time_deltas) > 0 else 1
            normalized_deltas = [delta / max_delta for delta in time_deltas]

            # Гарантируем целочисленный тип для buttonKey
            sequence_features = [
                [int(event['buttonKey']), float(normalized_deltas[i])]
                for i, event in enumerate(sequence)
            ]

            processed_sequences.append(sequence_features)

        # Паддинг с явным указанием типа float32
        X_padded = pad_sequences(
            processed_sequences,
            maxlen=self.max_sequence_length,
            dtype='float32',
            padding='post',
            truncating='post',
            value=0.0
        )

        y_array = np.array(y_raw, dtype='float32')

        return X_padded, y_array

    def optimize(self, X_train, y_train, n_trials=50, cv=3):
        """Оптимизация гиперпараметров модели с использованием Optuna"""
        print("Starting hyperparameter optimization with Optuna...")
        X_processed, y_processed = self._preprocess_data(X_train, y_train)
        
        # Разделение данных для кросс-валидации
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        def objective(trial):
            # Получение параметров из trial
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Кросс-валидация
            cv_scores = []
            for train_idx, val_idx in kf.split(X_processed, y_processed):
                X_cv_train, X_cv_val = X_processed[train_idx], X_processed[val_idx]
                y_cv_train, y_cv_val = y_processed[train_idx], y_processed[val_idx]
                
                # Построение модели с текущими параметрами
                model = self._build_model(trial)
                
                # Обучение модели
                model.fit(
                    X_cv_train, y_cv_train,
                    validation_data=(X_cv_val, y_cv_val),
                    epochs=5,  # Небольшое количество эпох для ускорения оптимизации
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
                )
                
                # Оценка модели
                _, accuracy = model.evaluate(X_cv_val, y_cv_val, verbose=0)
                cv_scores.append(accuracy)
                
                # Очистка сессии TensorFlow для предотвращения утечек памяти
                tf.keras.backend.clear_session()
            
            # Возвращаем среднюю точность по всем фолдам
            return np.mean(cv_scores)
        
        # Создание и запуск исследования Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Сохранение лучших параметров
        self.best_params = study.best_params
        
        # Вывод результатов оптимизации
        print("\nOptimization completed!")
        print(f"Best accuracy: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Построение финальной модели с лучшими параметрами
        self.model = self._build_model()
        
        return study.best_params

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=None):
        """Обучение модели с оптимальными гиперпараметрами"""
        if self.model is None:
            self.model = self._build_model()
            
        X_processed, y_processed = self._preprocess_data(X_train, y_train)

        if X_val is not None:
            X_val_processed, y_val_processed = self._preprocess_data(X_val, y_val)
            validation_data = (X_val_processed, y_val_processed)
        else:
            validation_data = None

        # Если batch_size не указан, используем оптимальный из best_params
        if batch_size is None and self.best_params and 'batch_size' in self.best_params:
            batch_size = self.best_params['batch_size']
        elif batch_size is None:
            batch_size = 32  # Значение по умолчанию

        history = self.model.fit(
            X_processed, y_processed,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        )
        return history

    def evaluate(self, X_test, y_test):
        """Оценка модели на тестовых данных"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        X_processed, y_processed = self._preprocess_data(X_test, y_test)
        loss, accuracy = self.model.evaluate(X_processed, y_processed, verbose=0)
        y_pred = (self.model.predict(X_processed) > 0.5).astype("int32")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_processed, y_pred))
        return y_pred

    def predict(self, X_new):
        """Предсказание на новых данных"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        X_processed, _ = self._preprocess_data(X_new, [0] * len(X_new))
        return (self.model.predict(X_processed) > 0.5).astype("int32")


# Пример использования
if __name__ == "__main__":
    # Загрузка данных
    print("Loading device logs...")
    sample_data = load_device_logs(1000)
    X, y = prepare_dataset(sample_data)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()

    # Инициализация модели
    print("Initializing GRU classifier...")
    classifier = GRUClassifier(max_sequence_length=50, num_button_keys=80)
    
    # Оптимизация гиперпараметров
    print("Starting hyperparameter optimization...")
    best_params = classifier.optimize(X_train, y_train, n_trials=20, cv=3)
    
    # Обучение модели с оптимальными параметрами
    print("\nTraining model with optimal hyperparameters...")
    classifier.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=15)
    
    end_train = time.time()
    training_time = end_train - start_train

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Оценка качества
    print("\nEvaluating model...")
    start_inf = time.time()
    y_pred = classifier.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    
    # Преобразование предсказаний в одномерный массив
    y_pred = [round(num) for sublist in y_pred for num in sublist]
    
    # Вывод метрик
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Training time: {training_time:.2f} s")
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Inference time: {inference_time:.4f} s")
