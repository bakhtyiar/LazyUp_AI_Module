import time
import tracemalloc
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from device_input.device_log_loader import load_device_logs


class ButtonPatternClassifier:
    def __init__(self, params=None):
        """Инициализация пайплайна с масштабированием и логистической регрессией"""
        if params is None:
            # Значения по умолчанию
            params = {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'max_iter': 1000,
                'tol': 1e-4,
                'fit_intercept': True,
                'l1_ratio': None,
                'random_state': 42
            }

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(**params)
        )

    def prepare_dataset(self, json_data):
        X = []
        y = []

        for item in json_data:
            X.append(item['list'])
            y.append(item['mode'])

        return X, y

    def _extract_features(self, dataset):
        """Извлечение признаков из сырых данных"""
        features = []
        for events in dataset:
            # Базовые статистики
            num_events = len(events)
            button_keys = [e['buttonKey'] for e in events]
            timestamps = [e['dateTime'] for e in events]

            # Временные характеристики
            time_diffs = []
            if len(timestamps) > 1:
                time_diffs = [
                    (timestamps[i] - timestamps[i + 1])
                    for i in range(len(timestamps) - 1)
                ]

                # Частотные характеристики кнопок
                button_counts = {i: 0 for i in range(1, 81)}
                for btn in button_keys:
                    if btn in button_counts:
                        button_counts[btn] += 1

                # Формирование вектора признаков
                feature_vec = {
                    'session_length': num_events,
                    'time_mean': np.mean(time_diffs) if time_diffs else 0,
                    'time_std': np.std(time_diffs) if time_diffs else 0,
                    **{f'btn{i}_ratio': button_counts[i] / num_events for i in range(1, 81)},
                    'rapid_clicks': sum(1 for diff in time_diffs if diff < 1.0)
                }
                features.append(feature_vec)

        return pd.DataFrame(features)

    def fit(self, X, y):
        """Обучение модели на размеченных данных"""
        X_features = self._extract_features(X)
        self.model.fit(X_features, y)
        return self

    def predict(self, X):
        """Предсказание классов для новых данных"""
        X_features = self._extract_features(X)
        return self.model.predict(X_features)

    def predict_proba(self, X):
        """Предсказание вероятностей классов"""
        X_features = self._extract_features(X)
        return self.model.predict_proba(X_features)

    def evaluate(self, X, y):
        """Оценка качества модели"""
        X_features = self._extract_features(X)
        y_pred = self.model.predict(X_features)
        return classification_report(y, y_pred, output_dict=True)


def objective(trial, X, y):
    """Функция цели для оптимизации Optuna"""
    # Определение параметров
    params = {
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None]),
        'C': trial.suggest_float('C', 1e-5, 100, log=True),
        'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
        'max_iter': trial.suggest_int('max_iter', 100, 2000),
        'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': 42,
        'l1_ratio': None  # значение по умолчанию
    }
    
    # Проверка и исправление несовместимых комбинаций
    
    # elasticnet работает только с saga
    if params['penalty'] == 'elasticnet':
        if params['solver'] != 'saga':
            params['solver'] = 'saga'
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
    
    # l1 работает только с liblinear и saga
    if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
        params['solver'] = 'saga'
    
    # None не работает с liblinear
    if params['penalty'] is None and params['solver'] == 'liblinear':
        params['solver'] = 'lbfgs'
    
    # Создание и оценка модели с помощью кросс-валидации
    classifier = ButtonPatternClassifier(params)
    X_features = classifier._extract_features(X)
    
    try:
        # Используем кросс-валидацию для более надежной оценки
        scores = cross_val_score(classifier.model, X_features, y, cv=5, scoring='accuracy')
        return scores.mean()
    except Exception as e:
        # В случае ошибки (например, из-за несовместимых параметров)
        print(f"Error with parameters {params}: {e}")
        return float('-inf')  # Возвращаем низкую оценку при ошибке


def optimize_hyperparameters(X, y, n_trials=100):
    """Оптимизация гиперпараметров с помощью Optuna"""
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study.best_params


# Пример использования
if __name__ == "__main__":
    sample_data = load_device_logs(1000)
    classifier = ButtonPatternClassifier()
    X_data, y_labels = classifier.prepare_dataset(sample_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

    # Оптимизация гиперпараметров
    print("Начало оптимизации гиперпараметров...")
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=50)

    # Обучение модели с лучшими параметрами
    print("\nОбучение модели с оптимальными параметрами...")
    tracemalloc.start()
    start_train = time.time()
    optimized_classifier = ButtonPatternClassifier(best_params)
    optimized_classifier.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train

    # Измерение памяти после обучения
    current, peak = tracemalloc.get_traced_memory()
    max_ram_usage = peak / (1024 ** 2)  # в MB
    tracemalloc.stop()

    # Оценка качества
    start_inf = time.time()
    y_pred = optimized_classifier.predict(X_test)
    end_inf = time.time()
    inference_time = end_inf - start_inf

    print("\nРезультаты оптимизированной модели:")
    print(classification_report(y_test, y_pred))
    print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
    print(f"Training time: {training_time:.4f} s")
    print(f"Inference time: {inference_time:.4f} s")