import os
import json
from datetime import datetime

def load_device_logs(max_records: int) -> list:
    """
    Загружает данные из файлов логов в папке device_input_logs,
    преобразует в агрегированный формат по режимам работы.

    Args:
        max_records: максимальное количество записей для загрузки

    Returns:
        список словарей с агрегированными данными по режимам:
        [
            {
                "mode": 0,
                "list": [
                    {"buttonKey": int, "dateTime": int (timestamp в мс)},
                    ...
                ]
            },
            {
                "mode": 1,
                "list": [
                    {"buttonKey": int, "dateTime": int (timestamp в мс)},
                    ...
                ]
            }
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

    # Инициализируем структуру для агрегированных данных
    aggregated_logs = {
        0: {"mode": 0, "list": []},
        1: {"mode": 1, "list": []}
    }

    total_records = 0

    # Обрабатываем файлы в обратном порядке (от новых к старым)
    for filename in reversed(log_files):
        if total_records >= max_records:
            break

        filepath = os.path.join(logs_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

                for entry in data.get("deviceLogs", []):
                    if total_records >= max_records:
                        break

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

                            aggregated_logs[mode]["list"].append(log_entry)
                            total_records += 1
                        except (ValueError, TypeError):
                            continue

        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при чтении файла {filename}: {e}")
            continue

    # Сортируем записи в каждом режиме по времени (от старых к новым)
    for mode_data in aggregated_logs.values():
        mode_data["list"].sort(key=lambda x: x["dateTime"])

    # Преобразуем словарь в список и возвращаем
    return list(aggregated_logs.values())

if __name__ == "__main__":
    json_formatted_str = json.dumps(load_device_logs(10), indent=2)
    print(json_formatted_str)
