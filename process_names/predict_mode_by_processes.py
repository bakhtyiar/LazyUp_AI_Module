import os
import datetime
import json

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from process_names.processes_log_loader import load_processes_logs

module_dir = Path(__file__).resolve().parent

directory_path = os.path.join(module_dir, 'processes_logs')  # Путь к директории с JSON-файлами
model_path = os.path.join(module_dir, 'predict_processes.joblib')  # Путь к модели

model = joblib.load(model_path)


def save_predictions_to_json(filename: str, predictions: np.ndarray, timestamps):
    """Save predictions and timestamps to JSON file"""
    data = {
        "predictions": [
            {"pred": float(pred), "timestamp": ts.isoformat()} 
            for pred, ts in zip(predictions, timestamps)
        ]
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def predict_by_processes(sample_data: list = None):
    """
        Args:
            список словарей с данными из каждого файла:
            [
                {
                    "mode": 0 | 1,
                    "list": [
                        {"buttonKey": int, "dateTime": int (timestamp в мс)},
                        ...
                    ]
                },
                ...
            ]

        Returns:
            ndarray с предсказаниями
            Array<0|1>
        """
    if sample_data is None:
        sample_data = load_processes_logs(10000)

    df = pd.DataFrame(sample_data)
    # Преобразуем процессы в строку (для CountVectorizer)
    df["processes_str"] = df["processes"].apply(lambda x: " ".join(x))
    # Разделяем на признаки (X) и целевую переменную (y)
    X = df[["timestamp", "processes_str"]]
    y_pred = model.predict(X)
    
    # Create prediction_logs directory if it doesn't exist
    prediction_logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prediction_logs')
    os.makedirs(prediction_logs_dir, exist_ok=True)
    
    # Get timestamps for predictions
    timestamps = [datetime.datetime.fromtimestamp(ts / 1000) for ts in df["timestamp"]]
    
    # Save predictions if we have sample data
    if sample_data and len(sample_data) > 0:
        first_datetime = timestamps[0]
        filename = os.path.join(prediction_logs_dir, f'predictions_{first_datetime.strftime("%Y%m%d_%H%M%S")}.json')
        save_predictions_to_json(filename, y_pred, timestamps)
    
    return y_pred


if __name__ == "__main__":
    ret = predict_by_processes(sample_data=load_processes_logs(1000))
