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

def save_processes_to_file(dir_to_logs, is_working_mode_target_value):
    processes = []
    processes_seen = set()  # Use a set to track seen (name, pid) tuples.
    for p in psutil.process_iter(attrs=['name', 'pid', 'cpu_percent']):
        if hasattr(p,'info'):
            try:
                # Получаем нагрузку на CPU
                p.cpu_percent(interval=0)  # Для получения текущего значения в будущем
                if p.info['name'] not in processes_seen:
                    processes_seen.add(p.info['name'])  # Mark this process as seen
                    processes.append((p.info['name'], p.info['pid'], p.cpu_percent(interval=0)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Сортируем процессы по нагрузке на CPU и по id процесса в порядке убывания
    sorted_processes = sorted(processes, key=lambda x: (x[2], x[1]), reverse=True)

    # Формируем массив с названиями процессов и нагрузкой на систему
    # result = [f"{name} (нагрузка: {cpu}% )" for name, pid, cpu in sorted_processes]
    timestamp = datetime.now().isoformat()
    result = {
        'is_working_mode': is_working_mode_target_value,
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
    logs_file_name = os.path.join(dir_to_logs, f"{clean_timestamp}.json")

    # Убеждаемся, что папка существует
    os.makedirs(dir_to_logs, exist_ok=True)

    # Записываем массив названий процессов в указанный файл
    print(logs_file_name)
    with open(logs_file_name, 'w') as json_file:
        # Записываем объект `result` в файл в формате JSON
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    print(f"Названия процессов с нагрузкой успешно записаны в {dir_to_logs}.")

def signal_handler(sig, frame):
    loggingInterval.cancel()
    print("Прерывание процесса. Сохранение данных и выход.")
    sys.exit(0)

if __name__ == "__main__":
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Записывает названия запущенных процессов в файлы JSON.')
    # Add an argument for the working mode
    parser.add_argument('--working-mode', type=str, default='false',
                        choices=['true', 'false', '1', '0', 'yes', 'no'],
                        help='Run in working mode (true/1/yes) or not (false/0/no).')
    parser.add_argument('--dir-to-log', type=str, default='.\processes_logs',
                        help='Имя выходного файла (по умолчанию processes_logs.json)')
    parser.add_argument('--log-interval-sec', type=str, default='5',
                        help='Как часто производить запись')
    # Parse the arguments
    args = parser.parse_args()
    is_working_mode = args.working_mode.lower() in ['true', '1', 'yes']
    logs_folder_name = args.dir_to_log
    time_interval = int(args.log_interval_sec)
    loggingInterval = set_interval(lambda: save_processes_to_file(logs_folder_name, is_working_mode), time_interval)
    signal.signal(signal.SIGINT, signal_handler)

