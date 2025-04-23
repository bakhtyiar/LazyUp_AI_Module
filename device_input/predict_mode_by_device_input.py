import os
import joblib
import numpy as np
from device_input.device_log_loader import load_device_logs
from device_input.train_model_mode_by_device_input import prepare_dataset

module_dir = os.path.dirname(os.path.abspath(__file__))

directory_path = module_dir + './device_input_logs'  # Путь к директории с JSON-файлами
model_path = module_dir + './predict_device_input.h5'  # Путь к модели

model = joblib.load(model_path)

def predict(sample_data):
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

    # Подготовка данных
    X, y = prepare_dataset(sample_data)

    y_pred = model.predict(X)
    return y_pred

def save_y_pred_to_file(filename: str, arr: np.ndarray):
    with open(filename or "y_pred.txt", 'a') as f:
        np.savetxt(f, arr, fmt='%.4f')

if __name__ == "__main__":
    ret = predict(sample_data=load_device_logs(1000))
    print(ret)
    save_y_pred_to_file("y_pred.txt", ret)