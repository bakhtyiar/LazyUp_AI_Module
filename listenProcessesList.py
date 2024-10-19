from pprint import pprint

import psutil
import json
import argparse
import pprint

def save_processes_to_file(filename):
    processes = []
    for p in psutil.process_iter(attrs=['name', 'pid', 'cpu_percent']):
        if hasattr(p,'info'):
            try:
                # Получаем нагрузку на CPU
                pprint.pprint(vars(p))
                p.cpu_percent(interval=0)  # Для получения текущего значения в будущем
                processes.append((p.info['name'], p.info['pid'], p.cpu_percent(interval=0)))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # Сортируем процессы по нагрузке на CPU и по id процесса в порядке убывания
    sorted_processes = sorted(processes, key=lambda x: (x[2], x[1]), reverse=True)

    # Формируем массив с названиями процессов и нагрузкой на систему
    # result = [f"{name} (нагрузка: {cpu}% )" for name, pid, cpu in sorted_processes]
    result = []
    for name, pid, cpu in sorted_processes:
        if name.endswith(".exe"):
            clean_process_name = name[:-4]  # Убираем последние 4 символа ".exe"
        else:
            clean_process_name = filename
        if not clean_process_name in result and not clean_process_name.endswith(".json"): # Не добавляем повторы и (json файлы?)
            result.append(clean_process_name)

    # Записываем массив названий процессов в указанный файл
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, ensure_ascii=False, indent=4)

    print(f"Названия процессов с нагрузкой успешно записаны в {filename}.")

if __name__ == "__main__":
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Записывает названия запущенных процессов в файл JSON.')
    parser.add_argument('filename', nargs='?', default='processes_logs.json',
                        help='Имя выходного файла (по умолчанию processes_logs.json)')

    # Получаем аргументы
    args = parser.parse_args()
    save_processes_to_file(args.filename)
