import os

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from process_names.processes_log_loader import load_processes_logs

module_dir = Path(__file__).resolve().parent

directory_path = os.path.join(module_dir, 'processes_logs')  # Путь к директории с JSON-файлами
model_path = os.path.join(module_dir, 'predict_processes.joblib')  # Путь к модели

model = joblib.load(model_path)


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
    return y_pred


def save_y_pred_to_file(filename: str, arr: np.ndarray):
    with open(filename or "y_pred.txt", 'w') as f:
        np.savetxt(f, arr, fmt='%.4f')


if __name__ == "__main__":
    ret = predict_by_processes(sample_data=load_processes_logs(1000))
    save_y_pred_to_file("y_pred.txt", ret)
