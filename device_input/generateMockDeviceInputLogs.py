import json
import random
import os
import datetime
import math


def generateTimestamp(base_time, index, total):
    """Генерация временных меток с нелинейным распределением"""
    # Нормализованная позиция в последовательности (0..1)
    pos = index / total

    # Применяем нелинейную функцию (синусоида + квадратичная)
    nonlinear_factor = 0.5 * math.sin(4 * math.pi * pos) + 0.5 * pos ** 2

    # Генерируем смещение в пределах 30 минут с нелинейным распределением
    max_seconds = 1800  # 30 минут
    offset_seconds = int(nonlinear_factor * max_seconds)

    return (base_time + datetime.timedelta(seconds=offset_seconds)).isoformat()


def generate_data_item(mode, base_time, index, total):
    timestamp = generateTimestamp(base_time, index, total)

    # Нормализованное время (0..1) для нелинейных зависимостей
    time_pos = index / total

    if mode == 1:
        is_working_mode = True
        # Квадратичная зависимость + синусоидальные колебания
        button_key = int(25 * (1 + math.sin(5 * time_pos)) + 10 * time_pos ** 2 + random.randint(-3, 3))
    elif mode == 2:
        is_working_mode = random.choice([True, False])
        # Экспоненциальная зависимость с шумом
        exp_factor = math.exp(2 * time_pos) - 1
        button_key = int(30 + 15 * math.sin(3 * time_pos) + 10 * exp_factor + random.randint(-5, 5))
    elif mode == 3:
        is_working_mode = False
        # Логарифмическая зависимость с переключениями
        log_factor = math.log1p(5 * time_pos)
        button_key = int(50 + 15 * math.sin(8 * time_pos) + 10 * log_factor + random.randint(-7, 7))

        # Ограничим значения в разумных пределах
        button_key = max(1, min(80, button_key))

    return {
        'timestamp': timestamp,
        'buttonKey': button_key,
        'isWorkingMode': is_working_mode
    }


def generate_device_logs(num_items):
    mode = random.randint(1, 3)
    base_time = datetime.datetime.now() - datetime.timedelta(minutes=30)

    # Сортируем элементы по времени после генерации
    items = [generate_data_item(mode, base_time, i, num_items) for i in range(num_items)]
    items.sort(key=lambda x: x['timestamp'])  # Сортировка по времени

    return {
        'deviceLogs': items
    }


currentDateTimeStamp = datetime.datetime.now() - datetime.timedelta(hours=0, minutes=0, seconds=5)

if __name__ == "__main__":
    num_items_key_inputs = 1000  # Количество нажатий клавиш
    num_sessions = 1000  # Количество сессий

    # Генерация и запись файлов
    for i in range(num_sessions):
        logs = generate_device_logs(num_items_key_inputs)

        file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        currentDateTimeStamp = currentDateTimeStamp - datetime.timedelta(hours=1)

        while os.path.exists(file_name):
            currentDateTimeStamp = currentDateTimeStamp - datetime.timedelta(hours=1)
            file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"

        logs_folder_name = "device_input_logs"
        logs_file_name = os.path.join(logs_folder_name, file_name)

        os.makedirs(logs_folder_name, exist_ok=True)

        with open(logs_file_name, 'w') as json_file:
            json.dump(logs, json_file, ensure_ascii=False, indent=4)