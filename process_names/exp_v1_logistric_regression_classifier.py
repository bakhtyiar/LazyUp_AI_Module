import time
import tracemalloc
import optuna
import numpy as np

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from process_names.processes_log_loader import load_processes_logs


def objective(trial):
    # Определяем все параметры независимо, а затем проверяем их совместимость
    penalty_options = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none'])
    solver_options = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    
    # Проверяем совместимость penalty и solver и корректируем при необходимости
    # Правила совместимости:
    # 1. penalty='elasticnet' только с solver='saga'
    # 2. penalty='l1' только с solver in ['liblinear', 'saga']
    # 3. penalty=None несовместим с solver='liblinear'
    
    # Корректировка несовместимых комбинаций
    if penalty_options == 'elasticnet' and solver_options != 'saga':
        solver_options = 'saga'
    elif penalty_options == 'l1' and solver_options not in ['liblinear', 'saga']:
        solver_options = trial.suggest_categorical('solver_l1', ['liblinear', 'saga'])
    elif penalty_options == 'none' and solver_options == 'liblinear':
        solver_options = trial.suggest_categorical('solver_none', ['newton-cg', 'lbfgs', 'sag', 'saga'])
    
    # Параметр l1_ratio нужен только при penalty='elasticnet'
    l1_ratio = None
    if penalty_options == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
    # Остальные параметры
    C = trial.suggest_float('C', 1e-5, 100.0, log=True)
    tol = trial.suggest_float('tol', 1e-5, 1e-2, log=True)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    intercept_scaling = trial.suggest_float('intercept_scaling', 0.1, 10.0)
    max_iter = trial.suggest_int('max_iter', 50, 1000)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    
    # Создаем пайплайн
    preprocessor = ColumnTransformer(
        transformers=[
            ('processes', CountVectorizer(binary=True), 'processes_str'),
            ('timestamp', StandardScaler(), ['timestamp']),
        ]
    )
    
    # Создаем модель с выбранными параметрами
    lr_params = {
        'penalty': penalty_options,
        'C': C,
        'solver': solver_options,
        'tol': tol,
        'fit_intercept': fit_intercept,
        'intercept_scaling': intercept_scaling,
        'max_iter': max_iter,
        'class_weight': class_weight,
        'random_state': 42
    }
    
    # Добавляем l1_ratio только если penalty='elasticnet'
    if l1_ratio is not None:
        lr_params['l1_ratio'] = l1_ratio
    
    try:
        model = make_pipeline(
            preprocessor,
            LogisticRegression(**lr_params)
        )
        
        # Используем кросс-валидацию для оценки модели
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        return np.mean(scores)
    except Exception as e:
        # В случае ошибки возвращаем плохой результат
        print(f"Error with parameters {lr_params}: {str(e)}")
        return float('-inf')


# Загружаем данные
data = load_processes_logs(1000)
df = pd.DataFrame(data)

# Преобразуем процессы в строку (для CountVectorizer)
df['processes_str'] = df['processes'].apply(lambda x: ' '.join(x))

# Разделяем на признаки (X) и целевую переменную (y)
X = df[['timestamp', 'processes_str']]
y = df['mode']

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем исследование Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Можно увеличить количество испытаний для лучших результатов

# Получаем лучшие параметры
best_params = study.best_params
print('Best parameters:', best_params)
print('Best f1_weighted score:', study.best_value)

# Создаем пайплайн с лучшими параметрами
preprocessor = ColumnTransformer(
    transformers=[
        ('processes', CountVectorizer(binary=True), 'processes_str'),
        ('timestamp', StandardScaler(), ['timestamp']),
    ]
)

# Подготавливаем параметры для лучшей модели
best_lr_params = {k: v for k, v in best_params.items() 
                 if k not in ['l1_ratio', 'solver_l1', 'solver_none']}

# Определяем правильный solver
if 'solver_l1' in best_params:
    best_lr_params['solver'] = best_params['solver_l1']
elif 'solver_none' in best_params:
    best_lr_params['solver'] = best_params['solver_none']

# Добавляем l1_ratio если нужно
if 'penalty' in best_params and best_params['penalty'] == 'elasticnet' and 'l1_ratio' in best_params:
    best_lr_params['l1_ratio'] = best_params['l1_ratio']

# Создаем модель с лучшими параметрами
best_model = make_pipeline(
    preprocessor,
    LogisticRegression(**best_lr_params, random_state=42)
)

# Измерение использования памяти до обучения
tracemalloc.start()
start_train = time.time()

# Обучаем модель
best_model.fit(X_train, y_train)

end_train = time.time()
training_time = end_train - start_train
# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

start_inf = time.time()
# Предсказываем на тестовых данных
y_pred = best_model.predict(X_test)
end_inf = time.time()
inference_time = end_inf - start_inf

# Оценка модели
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print(f'Max RAM Usage: {max_ram_usage:.2f} MB')
print(f'Training time: {training_time:.4f} s')
print(f'Inference time: {inference_time:.4f} s')

# # Пример предсказания для новых данных
# new_data = pd.DataFrame({
#     'timestamp': [123460],
#     'processes_str': ['chrome.exe python.exe']
# })
# prediction = best_model.predict(new_data)
# print('\nPrediction for new data:', prediction)
