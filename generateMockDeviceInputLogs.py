import json
import random
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


def generate_data_item():
    timestamp = generateTimestamp()
    kindOfRecord = random.randint(1, 3)
    if (kindOfRecord == 1):
        is_working_mode = True
        button_key = random.randint(1, 50)
    elif (kindOfRecord == 2):
        is_working_mode = random.choice([True, False])
        button_key = random.randint(30, 60)
    else:
        is_working_mode = False
        button_key = random.randint(50, 80)
    return {
        'timestamp': timestamp,
        'buttonKey': button_key,
        'isWorkingMode': is_working_mode
    }

def generate_device_logs(num_items):
    return {
        'deviceLogs': [generate_data_item() for _ in range(num_items)]
    }

def save_to_json(filename, data):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    num_items = 1000  # Укажите количество записей, которые нужно сгенерировать
    logs = generate_device_logs(num_items)
    save_to_json('device_logs.json', logs)