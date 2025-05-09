import json
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from files_cryptor import JsonFolderCrypto

# Основные процессы с категориями
mandatory_processes = {
    "SystemSettings": "system",
    "CompPkgSrv": "system",
    "SearchFilterHost": "search",
    "TextInputHost": "input",
    "CTMsgHostEdge": "communication",
    "ps64ldr": "system",
    "CCXProcess": "communication",
    "PhoneExperienceHost": "communication",
    "ServiceHub.Helper": "development",
    "SecurityHealthService": "security",
    "SecurityHealthSystray": "security",
    "msedgewebview2": "browser",
    "msedge": "browser",
    "RuntimeBroker": "system",
    "SearchProtocolHost": "search",
    "SgrmBroker": "security",
    "NisSrv": "security",
    "AggregatorHost": "system",
    "SDXHelper": "system",
    "dllhost": "system",
    "crashpad_handler": "utility",
    "SearchIndexer": "search"
}

# Добавочные процессы с категориями
joy_processes = {
    'steam': 'gaming',
    'discord': 'communication',
    'origin': 'gaming',
    'spotify': 'media',
    'vscode': 'development',
    'minecraft': 'gaming'
}

work_processes = {
    'whatsapp': 'communication',
    'telegram': 'communication',
    'word': 'office',
    'excel': 'office',
    'outlook': 'office',
    'teams': 'communication',
    'zoom': 'communication',
    'slack': 'communication',
    'postman': 'development',
    'git': 'development'
}

# Дополнительные параметры системы
system_metrics = {
    "cpu_usage": (5, 95),
    "memory_usage": (10, 90),
    "disk_activity": (0, 100),
    "network_activity": (0, 100),
    "gpu_usage": (0, 80),
    "temperature": (30, 80)
}


# Временные закономерности (день/ночь, рабочие часы)
def is_work_hours(timestamp):
    hour = timestamp.hour
    return 9 <= hour < 18  # Рабочие часы с 9 до 18


def is_weekday(timestamp):
    return timestamp.weekday() < 5  # Пн-Пт


# Временные закономерности для процессов
def get_time_based_probabilities(timestamp):
    hour = timestamp.hour
    is_night = hour < 6 or hour >= 23

    if is_work_hours(timestamp) and is_weekday(timestamp):
        # Рабочее время в рабочий день
        work_prob = 0.85
        joy_prob = 0.15
        system_load = random.uniform(0.6, 0.9)
    elif is_night:
        # Ночное время
        work_prob = 0.05
        joy_prob = 0.7
        system_load = random.uniform(0.2, 0.5)
    else:
        # Вечер/выходные
        work_prob = 0.3
        joy_prob = 0.65
        system_load = random.uniform(0.4, 0.7)

    return work_prob, joy_prob, system_load


# Зависимости между процессами
process_dependencies = {
    'word': ['excel', 'outlook'],
    'excel': ['word', 'outlook'],
    'outlook': ['word', 'excel'],
    'teams': ['outlook'],
    'zoom': ['outlook'],
    'steam': ['discord'],
    'vscode': ['git', 'postman'],
    'postman': ['vscode']
}


# Генерация метрик системы на основе активности
def generate_system_metrics(process_count, system_load):
    metrics = {}
    load_factor = system_load * random.uniform(0.9, 1.1)

    for metric, (min_val, max_val) in system_metrics.items():
        base_value = random.uniform(min_val, max_val) * load_factor
        # Учитываем количество процессов
        if metric in ['cpu_usage', 'memory_usage']:
            base_value = min(max_val, base_value + process_count * 0.5)
        metrics[metric] = round(base_value, 2)

    return metrics


# Функция для генерации данных
def generate_data(currentDateTimeStamp):
    work_prob, joy_prob, system_load = get_time_based_probabilities(currentDateTimeStamp)

    # Базовые процессы
    processes = list(mandatory_processes.keys())
    process_categories = defaultdict(int)

    # Определяем режим (работа/отдых) с учетом времени
    is_working_mode = random.random() < work_prob / (work_prob + joy_prob)

    # Добавляем рабочие или развлекательные процессы
    selected_processes = work_processes if is_working_mode else joy_processes
    main_prob = work_prob if is_working_mode else joy_prob
    secondary_prob = joy_prob if is_working_mode else work_prob

    # Основные процессы для текущего режима
    for proc, category in selected_processes.items():
        if random.random() < main_prob:
            processes.append(proc)
            process_categories[category] += 1

            # Добавляем зависимые процессы
            for dependent in process_dependencies.get(proc, []):
                if random.random() < 0.7:  # 70% вероятность зависимого процесса
                    processes.append(dependent)
                    dep_category = (work_processes | joy_processes).get(dependent, 'unknown')
                    process_categories[dep_category] += 1

    # Вторичные процессы (из противоположной категории)
    opposite_processes = work_processes if not is_working_mode else joy_processes
    for proc, category in opposite_processes.items():
        if random.random() < secondary_prob * 0.5:  # Уменьшенная вероятность
            processes.append(proc)
            process_categories[category] += 1

    # Генерация метрик системы
    metrics = generate_system_metrics(len(processes), system_load)

    # Добавляем некоторые системные процессы при высокой нагрузке
    if metrics['cpu_usage'] > 70 and random.random() < 0.7:
        processes.extend(['SystemSettings', 'RuntimeBroker', 'dllhost'])

    # Уникальные процессы и перемешивание
    processes = list(set(processes))
    random.shuffle(processes)

    return {
        "is_working_mode": is_working_mode,
        "timestamp": currentDateTimeStamp.isoformat(),
        "processes": processes,
        "process_categories": dict(process_categories),
        "system_metrics": metrics,
        "time_context": {
            "is_work_hours": is_work_hours(currentDateTimeStamp),
            "is_weekday": is_weekday(currentDateTimeStamp),
            "hour_of_day": currentDateTimeStamp.hour
        }
    }


# Настройка генерации
currentDateTimeStamp = datetime.now() - timedelta(minutes=5)
logs_folder_name = "processes_logs"
os.makedirs(logs_folder_name, exist_ok=True)

# Initialize encryption
crypto = JsonFolderCrypto()

for j in range(20):
    if j % 2 == 0:
        currentDateTimeStamp = currentDateTimeStamp - timedelta(minutes=600)
    currentDateTimeStamp = currentDateTimeStamp - timedelta(minutes=480)

    # Генерация и запись файлов
    for i in range(420):
        currentDateTimeStamp = currentDateTimeStamp - timedelta(minutes=1)
        data = generate_data(currentDateTimeStamp)
        file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"
        # Убедимся, что имена файлов уникальны
        while os.path.exists(os.path.join(logs_folder_name, file_name)):
            currentDateTimeStamp = currentDateTimeStamp - timedelta(minutes=1)
            file_name = currentDateTimeStamp.strftime("%Y-%m-%d_%H-%M-%S") + ".json"

        file_path = os.path.join(logs_folder_name, file_name)
        # First save the file normally
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        
        # Then encrypt it in place
        crypto.encrypt_file(file_path)
