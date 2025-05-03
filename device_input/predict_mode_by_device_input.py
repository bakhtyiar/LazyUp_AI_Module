import datetime
import json
import os
from pathlib import Path

import joblib
import numpy as np

from device_input.device_log_loader import load_device_logs
from device_input.train_model_mode_by_device_input import prepare_dataset

module_dir = Path(__file__).resolve().parent

directory_path = os.path.join(module_dir, 'device_input_logs')  # Путь к директории с JSON-файлами
prediction_logs_dir = os.path.join(module_dir, 'prediction_logs')
model_path = os.path.join(module_dir, 'predict_device_input.h5')  # Путь к модели

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


def predict_by_device_input(sample_data: list = None):
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
        sample_data = load_device_logs(10000)

    # Подготовка данных
    X, y = prepare_dataset(sample_data)

    y_pred = model.predict(X)

    # Create prediction_logs directory if it doesn't exist
    os.makedirs(prediction_logs_dir, exist_ok=True)

    # Get timestamps from sample data
    if sample_data and len(sample_data) > 0:
        timestamps = []
        for item in sample_data:
            if item.get('list') and len(item['list']) > 0:
                timestamps.append(
                    datetime.datetime.fromtimestamp(item['list'][0]['dateTime'] / 1000)
                )

        if timestamps:
            first_datetime = timestamps[0]
            filename = os.path.join(prediction_logs_dir,
                                    f'device_input_predictions_{first_datetime.strftime("%Y%m%d_%H%M%S")}.json')
            save_predictions_to_json(filename, y_pred, timestamps[:len(y_pred)])

    return y_pred


if __name__ == "__main__":
    ret = predict_by_device_input(sample_data=load_device_logs(5000))
