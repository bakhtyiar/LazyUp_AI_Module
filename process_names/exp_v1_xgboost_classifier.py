import time
import tracemalloc

import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

from process_names.processes_log_loader import load_processes_logs

# Пример данных (можно заменить на загрузку из JSON)
data = load_processes_logs()

# Преобразуем в DataFrame
df = pd.DataFrame(data)


# Извлечем признаки из processes (частоты, уникальные и т. д.)
def extract_features(df):
    # Количество процессов
    df['process_count'] = df['processes'].apply(len)

    # Временные признаки из timestamp - добавляем обработку ошибок
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    # Если есть невалидные даты, заменяем их на текущую
    df['datetime'] = df['datetime'].fillna(pd.Timestamp.now())
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # One-Hot Encoding для процессов
    all_processes = set(process for sublist in df['processes'] for process in sublist)
    for process in all_processes:
        df[f'process_{process}'] = df['processes'].apply(lambda x: 1 if process in x else 0)

    return df.drop(columns=['processes', 'timestamp', 'datetime'])


df = extract_features(df)
# Удаление ненужных столбцов
df = df.drop(['process_categories', 'system_metrics', 'time_context'], axis=1)
X = df.drop(columns=['mode'])
y = df['mode']

# Разделим данные на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    # Оптимизация всех параметров XGBoost
    # Сначала выбираем booster и tree_method, чтобы на их основе определить остальные параметры
    booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
    tree_method = trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist']) if booster != 'gblinear' else 'auto'
    
    params = {
        # Основные параметры
        'verbosity': 0,
        'booster': booster,
        'objective': 'multi:softmax',
        'num_class': len(set(y_train)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    }
    
    # Добавляем tree_method только для древовидных бустеров
    if booster != 'gblinear':
        params['tree_method'] = tree_method
        params['max_depth'] = trial.suggest_int('max_depth', 1, 20)
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 2000)
        params['gamma'] = trial.suggest_float('gamma', 0.001, 10.0, log=True)
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 0.1, 10.0, log=True)
        params['max_delta_step'] = trial.suggest_int('max_delta_step', 0, 10)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        
        # Добавляем colsample_bylevel только для совместимых методов
        params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
        
        # colsample_bynode только для hist и approx методов
        if tree_method not in ['exact']:
            params['colsample_bynode'] = trial.suggest_float('colsample_bynode', 0.5, 1.0)
    else:
        # Для линейного бустера
        params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
        params['n_estimators'] = trial.suggest_int('n_estimators', 50, 2000)
    
    # Параметры регуляризации для всех бустеров
    params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
    params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 10.0, log=True)
    
    # Дополнительные параметры для dart booster
    if booster == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 1.0)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 1.0)

    # Создаем модель с параметрами
    model = XGBClassifier(**params)
    
    # Используем кросс-валидацию для оценки модели
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
        scoring='f1_weighted'
    )
    return np.mean(scores)


# Создаем и запускаем исследование Optuna
print("\nЗапуск оптимизации гиперпараметров XGBoost...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Получаем лучшие параметры
best_params = study.best_params
print(f"\nЛучшие параметры: {best_params}")
print(f"Лучшее значение f1_weighted: {study.best_value:.4f}")

# Добавляем недостающие параметры для финальной модели
final_params = best_params.copy()
final_params['objective'] = 'multi:softmax'
final_params['num_class'] = len(set(y_train))
final_params['eval_metric'] = 'mlogloss'
final_params['use_label_encoder'] = False

# Создаем и обучаем модель с лучшими параметрами
best_model = XGBClassifier(**final_params)

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()

best_model.fit(X_train, y_train)

end_train = time.time()
training_time = end_train - start_train
# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

# Оценка
start_inf = time.time()
y_pred = best_model.predict(X_test)
end_inf = time.time()
inference_time = end_inf - start_inf

# Вывод результатов
print("\nРезультаты оценки модели:")
print(classification_report(y_test, y_pred))
print(f"\nВремя обучения: {training_time:.4f} с")
print(f"Использование памяти: {max_ram_usage:.2f} MB")
print(f"Время предсказания: {inference_time:.4f} с")

# Визуализация результатов оптимизации
try:
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    print("\nДля визуализации результатов используйте optuna.visualization")
except:
    print("\nДля визуализации результатов установите plotly: pip install plotly")

# Функция предсказания для новых данных
def predict_mode(new_data):
    # Преобразуем в DataFrame и извлекаем признаки
    new_df = pd.DataFrame([new_data])
    new_df = extract_features(new_df)
    
    # Убедимся, что все необходимые столбцы присутствуют
    for col in X_train.columns:
        if col not in new_df.columns:
            new_df[col] = 0
    
    # Убедимся, что порядок столбцов совпадает
    new_df = new_df[X_train.columns]
    
    # Предсказание
    prediction = best_model.predict(new_df)[0]
    
    return {"XGBoost (optimized)": prediction}
