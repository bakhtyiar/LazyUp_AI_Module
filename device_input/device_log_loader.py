import os
import json
from datetime import datetime

def load_device_logs(max_files: int) -> list:
    """
    Загружает данные из файлов логов в папке device_input_logs.
    Каждый файл представлен отдельным словарем в возвращаемом списке.

    Args:
        max_files: максимальное количество обработанных файлов

    Returns:
        список словарей с данными из каждого файла:
        [
            {
                "mode": 0 | 1,
                "list": [
                    {"buttonKey": int, "dateTime": int (timestamp в мс)},
                    ...
                ]
            },
            ...
        ]
    """
    log_files = []
    logs_dir = "./device_input_logs/"

    # Собираем все файлы логов из директории
    try:
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"Директория {logs_dir} не найдена")
        return []

    # Сортируем файлы по имени (по дате)
    log_files.sort()

    # Список для хранения данных из каждого файла
    file_logs = []
    total_records = 0

    # Обрабатываем файлы в обратном порядке (от новых к старым)
    for filename in reversed(log_files):
        if total_records >= max_files:
            break

        filepath = os.path.join(logs_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                file_data = {
                    "mode": int(data.get("deviceLogs", [])[0].get("isWorkingMode", False)),
                    "list": []
                }

                for entry in data.get("deviceLogs", []):
                    timestamp_str = entry.get("timestamp")
                    button_key = entry.get("buttonKey")
                    mode = int(entry.get("isWorkingMode", False))

                    if timestamp_str and button_key is not None:
                        try:
                            dt = datetime.fromisoformat(timestamp_str)
                            timestamp_ms = int(dt.timestamp() * 1000)

                            log_entry = {
                                "buttonKey": button_key,
                                "dateTime": timestamp_ms
                            }

                            file_data["list"].append(log_entry)
                        except (ValueError, TypeError):
                            continue

                # Сортируем записи в файле по времени (от старых к новым)
                file_data["list"].sort(key=lambda x: x["dateTime"])
                file_logs.append(file_data)
                total_records += 1

        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при чтении файла {filename}: {e}")
            continue

    return file_logs

if __name__ == "__main__":
    data = load_device_logs(10)
    json_formatted_str = json.dumps(data, indent=2)
    print(json_formatted_str)
