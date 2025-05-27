import os
from pathlib import Path

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from device_input.device_log_loader import load_device_logs
from device_input.exp_v1_cnn_classifier import preprocess_data, train_cnn_model

module_dir = Path(__file__).resolve().parent
model_path = os.path.join(module_dir, 'exp_v1_cnn_classifier.h5')


def train_model():
    """
    Обучает модель CNN на входных данных.

    Args:
        input_data: Данные в формате, возвращаемом load_device_logs
        model_path: Путь для сохранения модели

    Returns:
        Обученная модель Keras
    """
    return train_cnn_model()


def load_model(model_path=model_path):
    """
    Загружает обученную модель CNN из файла.

    Args:
        model_path: Путь к файлу модели

    Returns:
        Загруженная модель Keras
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Модель успешно загружена из {model_path}")
    return model


def predict_by_device_input(input_data, model=load_model(), max_sequence_length=50):
    """
    Делает предсказание режима на основе входных данных.

    Args:
        model: Загруженная модель Keras
        input_data: Данные в формате, возвращаемом load_device_logs
        max_sequence_length: Максимальная длина последовательности

    Returns:
        Предсказанные метки классов
    """
    # Предобработка данных
    X, _ = preprocess_data(input_data, max_sequence_length=max_sequence_length)

    # Нормализация данных (важно использовать те же параметры, что и при обучении)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Получение предсказаний
    predictions = model.predict(X_normalized)

    # Преобразование вероятностей в классы (для бинарной классификации)
    predicted_classes = [1 if pred >= 0.5 else 0 for pred in predictions.flatten()]

    return predicted_classes


if __name__ == "__main__":
    # Загрузка модели
    model = load_model(model_path)

    # Загрузка тестовых данных
    test_data = load_device_logs(10)  # Загружаем небольшой набор данных для тестирования

    # Получение предсказаний
    predicted_modes = predict_by_device_input(test_data, model)

    # Вывод результатов
    print("\nResults:")
    for i, (data, pred) in enumerate(zip(test_data, predicted_modes)):
        actual_mode = data['mode']
        print(f"{i + 1}: act: {actual_mode} pred: {pred}")

    # Вычисление точности на тестовых данных
    actual_modes = [item['mode'] for item in test_data]
    accuracy = sum(1 for a, p in zip(actual_modes, predicted_modes) if a == p) / len(actual_modes)
    print(f"\nAccuracy: {accuracy:.2f}")
