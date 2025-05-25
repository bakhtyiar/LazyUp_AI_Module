import time
import tracemalloc

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

from device_input.device_log_loader import load_device_logs


# Функция для извлечения признаков из сырых данных
def extract_features(data):
    """Преобразует сырые данные в DataFrame с признаками"""
    features = {
        'count': len(data),  # Общее количество нажатий
        'unique_buttons': len(set(d['buttonKey'] for d in data)),  # Уникальные кнопки
    }

    # Частотные признаки для кнопок
    button_counts = {}
    for d in data:
        button = d['buttonKey']
        button_counts[button] = button_counts.get(button, 0) + 1

    # Временные характеристики
    timestamps = [d['dateTime'] for d in data]  # Use timestamps directly
    timestamps.sort()

    if len(timestamps) > 1:
        time_diffs = [timestamps[i + 1] - timestamps[i]
                      for i in range(len(timestamps) - 1)]

        features.update({
            'time_mean': np.mean(time_diffs),
            'time_std': np.std(time_diffs),
            'time_max': max(time_diffs),
            'time_min': min(time_diffs),
            'time_median': np.median(time_diffs)
        })
        features['rapid_clicks'] = sum(1 for diff in time_diffs if diff < 2)
    else:
        # Default values when not enough timestamps
        features.update({
            'time_mean': 0,
            'time_std': 0,
            'time_max': 0,
            'time_min': 0,
            'time_median': 0,
            'rapid_clicks': 0
        })

    return pd.DataFrame([features])


# Подготовка датасета
def prepare_dataset(json_data):
    """Преобразует массив JSON объектов в обучающий датасет"""

    X = pd.DataFrame()
    y = []

    for item in json_data:
        features = extract_features(item['list'])
        X = pd.concat([X, features], ignore_index=True)
        y.append(item['mode'])

    # Заполнение пропусков (если временные признаки отсутствуют)
    X = X.fillna(0)

    return X, np.array(y)


# Функция для оптимизации гиперпараметров с Optuna
def optimize_xgboost_params(X, y, n_trials=100):
    """Использует Optuna для оптимизации всех параметров XGBoost"""
    
    # Определяем количество классов заранее
    num_classes = len(np.unique(y))
    print(f"Number of unique classes: {num_classes}")
    
    if num_classes <= 1:
        raise ValueError("Number of classes must be greater than 1 for classification")
    
    def objective(trial):
        # Определение всех параметров XGBoost для оптимизации
        params = {
            # Основные параметры
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            
            # Параметры регуляризации
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            
            # Параметры DART (только если booster='dart')
            'sample_type': trial.suggest_categorical('sample_type', ['uniform', 'weighted']),
            'normalize_type': trial.suggest_categorical('normalize_type', ['tree', 'forest']),
            'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
            'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
            
            # Прочие параметры
            'objective': 'multi:softmax',  # Для многоклассовой классификации
            'num_class': num_classes,  # Используем предварительно вычисленное значение
            'verbosity': 0,
            'random_state': 42
        }
        
        # Валидация параметров для DART бустера
        if params['booster'] != 'dart':
            # Удаляем параметры специфичные для DART
            for dart_param in ['sample_type', 'normalize_type', 'rate_drop', 'skip_drop']:
                params.pop(dart_param, None)
        
        # Оценка модели с помощью кросс-валидации
        model = XGBClassifier(**params)
        
        # Используем стратифицированную кросс-валидацию для сбалансированной оценки
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        return scores.mean()
    
    # Создаем и запускаем Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Получаем лучшие параметры
    best_params = study.best_params
    
    # Фильтруем параметры DART, если бустер не DART
    if best_params.get('booster') != 'dart':
        for dart_param in ['sample_type', 'normalize_type', 'rate_drop', 'skip_drop']:
            best_params.pop(dart_param, None) if dart_param in best_params else None
    
    print(f"Лучшие параметры: {best_params}")
    print(f"Лучшее значение f1_weighted: {study.best_value:.4f}")
    
    # Визуализация результатов оптимизации (при необходимости)
    try:
        print("\nВажность параметров:")
        importance = optuna.importance.get_param_importances(study)
        for key, value in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{key}: {value:.4f}")
    except:
        print("Не удалось рассчитать важность параметров")
    
    return best_params


# Обучение модели
def train_xgboost_model(X_train, y_train, params=None):
    """Обучает классификатор XGBoost с заданными параметрами"""
    # Определяем количество классов заранее
    num_classes = len(np.unique(y_train))
    print(f"Number of unique classes in training data: {num_classes}")
    
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'objective': 'multi:softmax',
            'num_class': num_classes
        }
    elif 'num_class' not in params and params.get('objective', '') in ['multi:softmax', 'multi:softprob']:
        # Если параметры переданы, но num_class отсутствует, добавляем его
        params['num_class'] = num_classes
    
    # Инициализация модели с оптимальными параметрами
    model = XGBClassifier(**params)
    # Обучение
    model.fit(X_train, y_train)
    return model


# Пример использования
if __name__ == "__main__":
    # Пример входных данных
    sample_data = load_device_logs(1000)

    # Подготовка данных
    X, y = prepare_dataset(sample_data)
    print("Извлеченные признаки:\n", X.head())

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Оптимизация гиперпараметров с Optuna (можно регулировать количество испытаний)
    best_params = optimize_xgboost_params(X_train, y_train, n_trials=50)

    # Измерение использования памяти до
    tracemalloc.start()
    start_train = time.time()

    # Обучение модели с оптимальными параметрами
    model = train_xgboost_model(X_train, y_train, best_params)
    end_train = time.time()
    training_time = end_train - start_train

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Оценка
    start_inf = time.time()
    y_pred = model.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Training time: {training_time:.4f} s")
    print(f"Inference time: {inference_time:.4f} s")
