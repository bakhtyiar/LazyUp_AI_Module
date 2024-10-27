import json
import random
import os
from datetime import datetime, timedelta
import time

# Основные процессы
mandatory_processes = [
    "SystemSettings", "CompPkgSrv", "SearchFilterHost", "TextInputHost",
    "CTMsgHostEdge", "ps64ldr", "CCXProcess", "PhoneExperienceHost",
    "ServiceHub.Helper", "SecurityHealthService", "SecurityHealthSystray",
    "msedgewebview2", "msedge", "RuntimeBroker", "SearchProtocolHost",
    "SgrmBroker", "NisSrv", "AggregatorHost", "SDXHelper", "dllhost",
    "crashpad_handler", "SearchIndexer"
]

# Добавочные процессы
joy_processes = ['steam', 'discord', 'origin']
work_processes = ['whatsapp', 'telegram', 'word', 'excel']

# Функция для генерации данных
def generate_data(currentDateTimeStamp):
    is_working_mode = random.choice([True, False])

    processes = mandatory_processes.copy()

    if is_working_mode:
        for proc in work_processes:
            if random.random() < 0.80:  # 80% вероятность
                processes.append(proc)
        for proc in joy_processes:
            if random.random() < 0.15:  # 15% вероятность
                processes.append(proc)
    else:
        for proc in work_processes:
            if random.random() < 0.15:  # 15% вероятность
                processes.append(proc)
        for proc in joy_processes:
            if random.random() < 0.80:  # 80% вероятность
                processes.append(proc)

    # timestamp = datetime.now().isoformat()

    processes.reverse()

    return {
        "is_working_mode": is_working_mode,
        "timestamp": currentDateTimeStamp.isoformat(),
        "processes": processes
    }

currentDateTimeStamp = datetime.now() - timedelta(hours=0, minutes=0, seconds=5)

# Генерация и запись файлов
for i in range(100):
    data = generate_data(currentDateTimeStamp)
    file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    currentDateTimeStamp = currentDateTimeStamp - timedelta(hours=0, minutes=0, seconds=5)
    
    # Убедимся, что имена файлов уникальны
    while os.path.exists(file_name):
        currentDateTimeStamp = currentDateTimeStamp - timedelta(hours=0, minutes=0, seconds=5)
        file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        # file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"

    logs_folder_name = "processes_logs"

    # Формируем имя файла в папке "logs_folder"
    logs_file_name = os.path.join(logs_folder_name, f"{file_name}.json")

    # Убеждаемся, что папка существует
    os.makedirs(logs_folder_name, exist_ok=True)

    with open(logs_file_name, 'w') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    # Задержка на секунду, чтобы избежать одинаковых временных меток
    # time.sleep(1)