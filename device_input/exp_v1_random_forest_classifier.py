import time
import tracemalloc

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score

from device_input.device_log_loader import load_device_logs


def extract_features(data):
    """Извлекает признаки из списка событий"""
    features = {}

    # Базовые признаки
    features['count'] = len(data)

    # Временные признаки
    timestamps = [x['dateTime'] for x in data]
    time_diffs = np.diff(timestamps)

    if len(time_diffs) > 0:
        features['time_diff_mean'] = np.mean(time_diffs)
        features['time_diff_std'] = np.std(time_diffs)
        features['time_diff_max'] = np.max(time_diffs)
        features['time_diff_min'] = np.min(time_diffs)
    else:
        features.update({
            'time_diff_mean': 0,
            'time_diff_std': 0,
            'time_diff_max': 0,
            'time_diff_min': 0
        })

    # Частотные признаки
    button_keys = [x['buttonKey'] for x in data]
    unique_keys, counts = np.unique(button_keys, return_counts=True)
    key_counts = dict(zip(unique_keys, counts))

    # Добавляем количество нажатий для каждой кнопки (до 10)
    for i in range(1, 11):
        features[f'key_{i}_count'] = key_counts.get(i, 0)

    # Относительные частоты
    total_presses = len(button_keys)
    for i in range(1, 11):
        features[f'key_{i}_ratio'] = features[f'key_{i}_count'] / total_presses if total_presses > 0 else 0

    # Временные паттерны
    if len(time_diffs) > 0:
        features['fast_events_5s'] = np.sum(np.array(time_diffs) <= 5)
        features['fast_events_10s'] = np.sum(np.array(time_diffs) <= 10)
    else:
        features.update({
            'fast_events_5s': 0,
            'fast_events_10s': 0
        })

    return features


def prepare_dataset(json_data):
    """Подготавливает датасет из сырых JSON данных"""
    X = []
    y = []

    for item in json_data:
        # Извлекаем целевой признак
        y.append(item['mode'])

        # Извлекаем признаки из списка событий
        features = extract_features(item['list'])
        X.append(features)

    # Преобразуем в DataFrame
    feature_df = pd.DataFrame(X)
    return feature_df, np.array(y)


def objective(trial, X, y):
    """Objective function for Optuna optimization"""
    # Define the hyperparameter space for RandomForestClassifier
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 100, log=True) if trial.suggest_categorical('use_max_leaf_nodes', [True, False]) else None,
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
        'random_state': 42,
        'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1),
        'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.params['bootstrap'] else None
    }
    
    # Handle conditional parameters
    if params['bootstrap'] is False:
        params['max_samples'] = None
    
    if trial.params.get('use_max_leaf_nodes', False) is False:
        params['max_leaf_nodes'] = None
    
    # Create and evaluate model using cross-validation
    model = RandomForestClassifier(**params)
    return cross_val_score(model, X, y, scoring='f1_weighted', cv=5).mean()


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)

    # Подготовка данных
    X, y = prepare_dataset(sample_data)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Оптимизация гиперпараметров с помощью Optuna
    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    
    # Получение лучших параметров
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best F1 score: {study.best_value:.4f}")
    
    # Если use_max_leaf_nodes параметр не был выбран, уберем max_leaf_nodes из параметров
    if 'use_max_leaf_nodes' in best_params and not best_params['use_max_leaf_nodes']:
        best_params['max_leaf_nodes'] = None
    if 'use_max_leaf_nodes' in best_params:
        del best_params['use_max_leaf_nodes']
    
    # Если bootstrap=False, установим max_samples=None
    if 'bootstrap' in best_params and best_params['bootstrap'] is False:
        best_params['max_samples'] = None
    
    # Измерение использования памяти до обучения
    tracemalloc.start()
    start_train = time.time()
    
    # Обучение модели с лучшими параметрами
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    
    end_train = time.time()
    training_time = end_train - start_train
    
    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Предсказание на тестовых данных с замером времени
    start_inf = time.time()
    y_pred = model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    
    # Оценка качества
    print("\nFinal model evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Training time: {training_time:.4f} s")
    print(f"Inference time: {inference_time:.4f} s")
