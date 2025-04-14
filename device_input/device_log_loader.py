import os
import json
from datetime import datetime

def load_device_logs(max_records: int, isWorkingMode: bool) -> dict:
    """
    Загружает данные из файлов логов в папке device_input_logs,
    фильтрует по режиму (isWorkingMode) и преобразует в требуемый формат.

    Args:
        max_records: максимальное количество записей для загрузки
        isWorkingMode: режим работы (True - рабочий, False - нерабочий)

    Returns:
        Словарь с данными в требуемом формате:
        {
            "mode": 1 (для рабочего) или 0 (для нерабочего),
            "list": [
                {"buttonKey": int, "dateTime": int (timestamp в мс)},
                ...
            ]
        }
    """
    log_files = []
    logs_dir = "./device_input_logs/"

    # Собираем все файлы логов из директории
    try:
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"Директория {logs_dir} не найдена")
        return {"mode": int(isWorkingMode), "list": []}

    # Сортируем файлы по имени (по дате)
    log_files.sort()

    result = {
        "mode": int(isWorkingMode),
        "list": []
    }

    # Обрабатываем файлы в обратном порядке (от новых к старым)
    for filename in reversed(log_files):
        if len(result["list"]) >= max_records:
            break

        filepath = os.path.join(logs_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

                for entry in data.get("deviceLogs", []):
                    if len(result["list"]) >= max_records:
                        break

                    if entry.get("isWorkingMode", None) == isWorkingMode:
                        timestamp_str = entry.get("timestamp")
                        button_key = entry.get("buttonKey")

                        if timestamp_str and button_key is not None:
                            try:
                                dt = datetime.fromisoformat(timestamp_str)
                                timestamp_ms = int(dt.timestamp() * 1000)

                                result["list"].append({
                                    "buttonKey": button_key,
                                    "dateTime": timestamp_ms
                                })
                            except (ValueError, TypeError):
                                continue

        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при чтении файла {filename}: {e}")
            continue

    # Сортируем записи по времени (от старых к новым)
    result["list"].sort(key=lambda x: x["dateTime"])

    return result

#
# if __name__ == "__main__":
#     print(load_device_logs(10, True))
#     print(load_device_logs(10, False))