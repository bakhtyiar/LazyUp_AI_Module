import json
import os
from datetime import datetime
from pathlib import Path

from logs_cypher import JsonFolderCrypto

module_dir = Path(__file__).resolve().parent
crypto = JsonFolderCrypto()


def load_device_logs(max_files: int | None = None, max_units: int | None = None) -> list:
    """
    Загружает данные из файлов логов в папке device_input_logs.
    Поддерживает как зашифрованные, так и незашифрованные JSON файлы.

    Args:
        max_units: максимальное количество залогированных действий
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
    logs_dir = os.path.join(module_dir, 'device_input_logs')

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
    total_units = 0

    # Обрабатываем файлы в обратном порядке (от новых к старым)
    for filename in reversed(log_files):
        if max_files is not None and total_records >= max_files:
            break

        if max_units is not None and total_units >= max_units:
            break

        filepath = os.path.join(logs_dir, filename)
        try:
            # Проверяем, зашифрован ли файл
            is_encrypted = crypto.is_file_encrypted(filepath)

            if is_encrypted:
                # Создаем временный файл для расшифрованных данных
                temp_filepath = filepath + '.temp'
                crypto.decrypt_file(filepath, temp_filepath)

                with open(temp_filepath, 'r') as f:
                    data = json.load(f)

                # Удаляем временный файл
                os.remove(temp_filepath)
            else:
                with open(filepath, 'r') as f:
                    data = json.load(f)

            # Calculate average mode from all valid entries
            valid_modes = []
            for entry in data.get("deviceLogs", []):
                # Skip entries without isWorkingMode key or with None value
                if "isWorkingMode" in entry and entry["isWorkingMode"] is not None:
                    # Convert True to 1, False to 0
                    mode_value = 1 if entry["isWorkingMode"] else 0
                    valid_modes.append(mode_value)

            # Calculate the average if there are valid modes, otherwise default to 0
            file_data = {
                "mode": round(sum(valid_modes) / len(valid_modes)) if valid_modes else 0,
                "list": []
            }

            for entry in data.get("deviceLogs", []):
                timestamp_str = entry.get("timestamp")
                button_key = entry.get("buttonKey")
                mode = int(entry.get("isWorkingMode", False))

                if max_units is not None and total_units >= max_units:
                    break

                if timestamp_str and button_key is not None:
                    try:
                        # Skip entries without isWorkingMode key or with None value
                        if "isWorkingMode" not in entry or entry["isWorkingMode"] is None:
                            continue

                        dt = datetime.fromisoformat(timestamp_str)
                        timestamp_ms = int(dt.timestamp() * 1000)

                        log_entry = {
                            "buttonKey": button_key,
                            "dateTime": timestamp_ms
                        }

                        file_data["list"].append(log_entry)
                        total_units += 1
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
    data = load_device_logs(10, 10000)
    json_formatted_str = json.dumps(data, indent=2)
    print(json_formatted_str)
