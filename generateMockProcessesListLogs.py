import json
import random
import os
from datetime import datetime, timedelta

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
def generate_data():
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

    timestamp = datetime.now().isoformat()

    return {
        "is_working_mode": is_working_mode,
        "timestamp": timestamp,
        "processes": processes
    }

currentDateTimeStamp = datetime.now() - timedelta(hours=0, minutes=0, seconds=5)

# Генерация и запись файлов
for i in range(100):
    data = generate_data()
    file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    currentDateTimeStamp = currentDateTimeStamp - timedelta(hours=0, minutes=0, seconds=5)
    
    # Убедимся, что имена файлов уникальны
    while os.path.exists(file_name):
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"

    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # Задержка на секунду, чтобы избежать одинаковых временных меток
    time.sleep(1)

# Генерация и запись файлов
for i in range(100):
    # Вычисляем название файла
    adjusted_timestamp = current_timestamp - (5 * i)
    file_name = datetime.datetime.fromtimestamp(adjusted_timestamp).strftime('%Y-%m-%d_%H-%M-%S') + '.json'
    
    # Создаем файл с указанным именем
    with open(file_name, 'w') as f:
        f.write(f'Файл создан с временной меткой: {adjusted_timestamp}\n')

print(f'Создано {num_files} файлов.')