import time
import tracemalloc

import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

from process_names.processes_log_loader import load_processes_logs

# Загрузка данных
data = load_processes_logs(1000)

# Преобразование в DataFrame
df = pd.DataFrame(data)
print(df.head())

mlb = MultiLabelBinarizer()
process_features = pd.DataFrame(mlb.fit_transform(df['processes']), columns=mlb.classes_)

# Объединение с исходными данными
df_processed = pd.concat([df.drop('processes', axis=1), process_features], axis=1)
print(df_processed.head())

# Удаление ненужных столбцов
df_processed = df_processed.drop(['process_categories', 'system_metrics', 'time_context'], axis=1)

print(df_processed.head())

X = df_processed.drop('mode', axis=1)  # Все признаки, кроме целевой переменной
y = df_processed['mode']  # Целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    # Основные параметры
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 50, log=True)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.5)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.01, 0.5)
    
    # Параметры для выбора признаков
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Параметры для бутстрэпа
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    
    # Параметры для критерия разделения
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    
    # Дополнительные параметры
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000, log=True) if trial.suggest_categorical('use_max_leaf_nodes', [True, False]) else None
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.1)
    
    # Параметры для класса весов
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample'])
    
    # Параметр max_samples только применим, когда bootstrap=True
    max_samples = None
    if bootstrap:
        max_samples = trial.suggest_float('max_samples', 0.5, 1.0)
    
    # Создание модели с выбранными параметрами
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        criterion=criterion,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha,
        class_weight=class_weight,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1
    )
    
    # Используем кросс-валидацию для более надежной оценки
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    return scores.mean()


# Создание исследования Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Можно увеличить количество испытаний для лучших результатов

# Получение лучших параметров
best_params = study.best_params
print(f"Best parameters: {best_params}")
print(f"Best f1 score: {study.best_value:.4f}")

# Создание модели с лучшими параметрами
# Обработка условных параметров
if not best_params.get('bootstrap', True):
    best_params.pop('max_samples', None)

if not best_params.get('use_max_leaf_nodes', False):
    best_params.pop('max_leaf_nodes', None)

# Удаление вспомогательного параметра
best_params.pop('use_max_leaf_nodes', None)

# Создание оптимальной модели
best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

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

# Оценка качества
sample_data = X_test
start_inf = time.time()

y_pred = best_model.predict(X_test)

end_inf = time.time()
inference_time = end_inf - start_inf
print(classification_report(y_test, y_pred))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Training time: {training_time:.4f} s")
print(f"Inference time: {inference_time:.4f} s")
