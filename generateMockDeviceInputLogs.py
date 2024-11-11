import json
import random
import os
import datetime

def generateTimestamp():
    # Получаем текущее время
    now = datetime.datetime.now()
    # Получаем время 30 минут назад
    thirty_minutes_ago = now - datetime.timedelta(minutes=30)
    # Генерируем случайное количество секунд в пределах 30 минут
    random_seconds = random.randint(0, 1800)  # от 0 до 1800 секунд
    # Вычисляем случайное время
    random_time = thirty_minutes_ago + datetime.timedelta(seconds=random_seconds)
    # Выводим результат в формате ISO
    return random_time.isoformat()


def generate_data_item(mode):
    timestamp = generateTimestamp()
    if mode == 1:
        is_working_mode = True
        button_key = random.randint(1, 50)
    elif mode == 2:
        is_working_mode = random.choice([True, False])
        button_key = random.randint(30, 60)
    elif mode == 3:
        is_working_mode = False
        button_key = random.randint(50, 80)
    return {
        'timestamp': timestamp,
        'buttonKey': button_key,
        'isWorkingMode': is_working_mode
    }

def generate_device_logs(num_items):
    mode = random.randint(1, 3)
    return {
        'deviceLogs': [generate_data_item(mode) for _ in range(num_items)]
    }

currentDateTimeStamp = datetime.datetime.now() - datetime.timedelta(hours=0, minutes=0, seconds=5)

if __name__ == "__main__":
    num_items_key_inputs = 1000  # Укажите количество нажатий клавиш, которые нужно сгенерировать
    num_sessions = 100  # Укажите количество сессий записей, которые нужно сгенерировать
    # Генерация и запись файлов
    for i in range(num_sessions):
        logs = generate_device_logs(num_items_key_inputs)

        file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        currentDateTimeStamp = currentDateTimeStamp - datetime.timedelta(hours=1, minutes=0, seconds=0)

        # Убедимся, что имена файлов уникальны
        while os.path.exists(file_name):
            currentDateTimeStamp = currentDateTimeStamp - datetime.timedelta(hours=1, minutes=0, seconds=0)
            file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
            # file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"

        logs_folder_name = "device_input_logs"

        # Формируем имя файла в папке "logs_folder"
        logs_file_name = os.path.join(logs_folder_name, f"{file_name}")

        # Убеждаемся, что папка существует
        os.makedirs(logs_folder_name, exist_ok=True)

        with open(logs_file_name, 'w') as json_file:
            json.dump(logs, json_file, ensure_ascii=False, indent=4)
