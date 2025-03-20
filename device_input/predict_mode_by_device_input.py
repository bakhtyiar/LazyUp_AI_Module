import json
import os
import pandas as pd
import joblib

module_dir = os.path.dirname(os.path.abspath(__file__))

directory_path = module_dir + './device_input_logs'  # Путь к директории с JSON-файлами
model_path = module_dir + './predict_device_input.h5'  # Путь к модели

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

X = prepared_data[['timestamp_seconds', 'buttonKey']]
y = prepared_data['isWorkingMode']

# Оценка модели
predictions = model.predict(X)

