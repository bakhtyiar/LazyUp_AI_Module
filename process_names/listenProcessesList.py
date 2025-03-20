import psutil
import json
import argparse
from datetime import datetime
import threading
import signal
import sys
import os
import time

is_working_mode = True

def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t

def save_processes_to_file(logs_folder_name, is_working_mode):
    processes = []
    processesSeen = set()  # Use a set to track seen (name, pid) tuples.
    for p in psutil.process_iter(attrs=['name', 'pid', 'cpu_percent']):
        if hasattr(p,'info'):
            try:
                # Получаем нагрузку на CPU
                p.cpu_percent(interval=0)  # Для получения текущего значения в будущем
                if p.info['name'] not in processesSeen:
                    processesSeen.add(p.info['name'])  # Mark this process as seen
                    processes.append((p.info['name'], p.info['pid'], p.cpu_percent(interval=0)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Сортируем процессы по нагрузке на CPU и по id процесса в порядке убывания
    sorted_processes = sorted(processes, key=lambda x: (x[2], x[1]), reverse=True)

    # Формируем массив с названиями процессов и нагрузкой на систему
    # result = [f"{name} (нагрузка: {cpu}% )" for name, pid, cpu in sorted_processes]
    timestamp = datetime.now().isoformat()
    result = {
        'is_working_mode': is_working_mode,
        'timestamp': timestamp,
        'processes': []
    }
    for name, pid, cpu in sorted_processes:
        if name.endswith(".exe"):
            clean_process_name = name[:-4]  # Убираем последние 4 символа ".exe"
        else:
            clean_process_name = name
        if not clean_process_name in result and not clean_process_name.endswith(".json"): # Не добавляем повторы и (json файлы?)
            result['processes'].append(clean_process_name)

    # Формируем имя файла в папке "logs_folder"
    clean_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") # Убраны неправильные символы для названия файла в windows
    logs_file_name = os.path.join(logs_folder_name, f"{clean_timestamp}.json")

    # Убеждаемся, что папка существует
    os.makedirs(logs_folder_name, exist_ok=True)

    # Записываем массив названий процессов в указанный файл
    print(logs_file_name)
    with open(logs_file_name, 'w') as json_file:
        # Записываем объект `result` в файл в формате JSON
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    print(f"Названия процессов с нагрузкой успешно записаны в {logs_folder_name}.")

def signal_handler(sig, frame):
    loggingInterval.cancel()
    print("Прерывание процесса. Сохранение данных и выход.")
    sys.exit(0)

if __name__ == "__main__":
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Записывает названия запущенных процессов в файл JSON.')
    if len(sys.argv) > 1:
        is_working_mode = sys.argv[1].lower() in ['true', '1', 'yes']
    # parser.add_argument('logs_folder_name', nargs='?', default='processes_logs',
    #                     help='Имя выходного файла (по умолчанию processes_logs.json)')
    logs_folder_name = "processes_logs"

    # Получаем аргументы
    args = parser.parse_args()
    # save_processes_to_file(args.logs_folder_name)
    # loggingInterval = set_interval(lambda: save_processes_to_file(args.logs_folder_name), 5)
    loggingInterval = set_interval(lambda: save_processes_to_file(logs_folder_name, is_working_mode), 5)
    signal.signal(signal.SIGINT, signal_handler)

