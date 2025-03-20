import json
import os
import pandas as pd
import joblib

module_dir = os.path.dirname(os.path.abspath(__file__))

directory_path = module_dir + './device_input_logs'  # Путь к директории с JSON-файлами
model_path = module_dir + './predict_device_input.h5'  # Путь к модели

def predict_by_device_input(df: pd.DataFrame):
    # Преобразуем timestamp в числовой формат
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_seconds'] = df['timestamp'].astype('int64') // 10**9  # Конвертация в секунды
    # Сортировка по timestamp
    df = df.sort_values('timestamp')

    # Выбираем необходимые столбцы
    df = df[['timestamp_seconds', 'buttonKey', 'isWorkingMode']]
    prepared_data = df

    model = joblib.load(model_path)

    X = prepared_data[['timestamp_seconds', 'buttonKey']]
    y = prepared_data['isWorkingMode']

    # Оценка модели
    ret = model.predict(X)
    return ret

def load_dataframe_device_input(amount_of_records=128):
    # Считываем все логи
    all_logs = []
    for filename in os.listdir(directory_path):
        if amount_of_records < 1:
            break
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                all_logs.extend(data['deviceLogs'])
        amount_of_records -= 1

    # Подготовка данных
    df = pd.DataFrame(all_logs)
    return df

def predict():
    df = load_dataframe_device_input(amount_of_records=128)
    return predict_by_device_input(df)

if __name__ == "__main__":
    ret = predict()
    json.dumps(ret)