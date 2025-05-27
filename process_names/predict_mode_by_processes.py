import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from process_names.exp_v1_logistric_regression_classifier import train_logistic_regression_model
from process_names.processes_log_loader import load_processes_logs

module_dir = Path(__file__).resolve().parent
model_path = os.path.join(module_dir, 'predict_processes.joblib')
model_params_path = os.path.join(module_dir, 'predict_processes_params.joblib')


def train_model():
    """
        Обучает модель CNN на входных данных.

        Args:
            input_data: Данные в формате, возвращаемом load_device_logs
            model_path: Путь для сохранения модели

        Returns:
            Обученная модель Keras
        """
    return train_logistic_regression_model()


def load_model(model_path=model_path):
    """
    Загружает обученную модель CNN из файла.

    Args:
        model_path: Путь к файлу модели

    Returns:
        Загруженная модель Keras
    """
    model = joblib.load(model_path)
    print(f"Модель успешно загружена из {model_path}")
    return model


def predict_by_processes(processes=load_processes_logs(), timestamp=datetime.now().isoformat(), model=load_model()):
    """
    Predict mode using trained model.
    
    Args:
        processes (list): List of process names
        timestamp (int): Timestamp value
        model: Загруженная модель
        max_sequence_length: Максимальная длина последовательности
    Returns:
        str: Predicted mode
    """
    # Load the model
    if not os.path.exists(model):
        raise FileNotFoundError(f"Model file not found at {model}. Train the model first.")

    # Use joblib.load instead of pickle.load
    model = load_model()

    # Prepare input data
    processes_str = ' '.join(processes)
    data = pd.DataFrame({
        'timestamp': [timestamp],
        'processes_str': [processes_str]
    })

    # Make prediction
    prediction = model.predict(data)
    return prediction[0]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Predict mode based on running processes')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--model_path', type=str, default='model.pkl', help='Path to model file')
    parser.add_argument('--processes', type=str, nargs='+', help='List of process names')
    parser.add_argument('--timestamp', type=int, help='Timestamp value')
    parser.add_argument('--sample_size', type=int, default=1000, help='Sample size for training')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')

    args = parser.parse_args()

    if args.train:
        print(f"Training model with {args.sample_size} samples and {args.n_trials} trials...")
        train_model(sample_size=args.sample_size, n_trials=args.n_trials, model_path=args.model_path)
    elif args.processes and args.timestamp is not None:
        try:
            mode = predict_by_processes(args.processes, args.timestamp, model=args.model_path)
            print(f"Predicted mode: {mode}")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
