import json
import os
from typing import List, Dict, Any
from datetime import datetime


def load_processes_logs(max_files: int = 100, directory: str = "./processes_logs") -> List[Dict[str, Any]]:
    """
    Загружает данные из всех JSON-файлов в указанной директории,
    заменяя ключ 'is_working_mode' на 'mode'.

    Args:
        directory (str): Путь к директории с лог-файлами. По умолчанию "./processes_logs".

    Returns:
        List[Dict[str, Any]]: Список словарей с данными из лог-файлов.
    """
    logs_data = []
    total_records = 0

    # Проверяем существование директории
    if not os.path.exists(directory):
        print(f"Директория {directory} не существует!")
        return logs_data

    # Проходим по всем файлам в директории
    for filename in os.listdir(directory):
        if total_records >= max_files:
            break
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                    # Заменяем 'is_working_mode' на 'mode', если ключ существует
                    if "is_working_mode" in data:
                        data["mode"] = data.pop("is_working_mode")

                    timestamp_str = data.get("timestamp")
                    if timestamp_str is not None:
                        try:
                            dt = datetime.fromisoformat(timestamp_str)
                            timestamp_ms = int(dt.timestamp() * 1000)
                            data["timestamp"] = timestamp_ms

                        except (ValueError, TypeError):
                            continue


                    logs_data.append(data)
            except json.JSONDecodeError:
                print(f"Ошибка при чтении файла {filename}: файл не является валидным JSON")
            except Exception as e:
                print(f"Ошибка при чтении файла {filename}: {str(e)}")

    return logs_data


# Пример использования
if __name__ == "__main__":
    logs = load_processes_logs(1000)
    for log in logs:
        print(log)