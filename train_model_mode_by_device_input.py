import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

directory_path = './device_input_logs'  # Путь к директории с JSON-файлами
model_path = './predict_device_input.h5'  # Путь к модели

# Считываем все логи
all_logs = []
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            all_logs.extend(data['deviceLogs'])

# Подготовка данных
df = pd.DataFrame(all_logs)

# Преобразуем timestamp в числовой формат
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp_seconds'] = df['timestamp'].astype('int64') // 10**9  # Конвертация в секунды

# Выбираем необходимые столбцы
df = df[['timestamp_seconds', 'buttonKey', 'isWorkingMode']]
prepared_data = df

model = None
try:
    model = joblib.load(model_path)
except OSError:
    print("Saved model not found")

# Обучение модели
# Разделение данных на признаки и целевую переменную
X = prepared_data[['timestamp_seconds', 'buttonKey']]
y = prepared_data['isWorkingMode']

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if not model:
    # Создание и обучение модели
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # model.save(model_path)
    joblib.dump(model, model_path)

# Оценка модели
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

