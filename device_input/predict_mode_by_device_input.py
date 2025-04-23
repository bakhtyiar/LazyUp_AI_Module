import os
import joblib
import numpy as np
from device_input.device_log_loader import load_device_logs
from device_input.train_model_mode_by_device_input import prepare_dataset
from pathlib import Path

module_dir = Path(__file__).resolve().parent

directory_path = os.path.join(module_dir, 'device_input_logs')  # Путь к директории с JSON-файлами
model_path = os.path.join(module_dir, 'predict_device_input.h5')  # Путь к модели

model = joblib.load(model_path)

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
    return y_pred

def save_y_pred_to_file(filename: str, arr: np.ndarray):
    with open(filename or "y_pred.txt", 'w') as f:
        np.savetxt(f, arr, fmt='%.4f')

if __name__ == "__main__":
    ret = predict_by_device_input(sample_data=load_device_logs(1000))
    save_y_pred_to_file("y_pred.txt", ret)