import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from logs_cypher import JsonFolderCrypto

module_dir = Path(__file__).resolve().parent
crypto = JsonFolderCrypto()


def load_processes_logs(max_files: int = 100, directory: str = os.path.join(module_dir, 'processes_logs')) -> List[
    Dict[str, Any]]:
    """
    Загружает данные из всех JSON-файлов в указанной директории,
    поддерживает как зашифрованные, так и незашифрованные файлы.
    Заменяет ключ 'is_working_mode' на 'mode'.
    Пропускает файлы, в которых отсутствует ключ 'is_working_mode'.

    Args:
        max_files (int): Максимальное количество обрабатываемых файлов. По умолчанию 100.
        directory (str): Путь к директории с лог-файлами. По умолчанию "processes_logs".

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
                # Проверяем, зашифрован ли файл
                is_encrypted = crypto.is_file_encrypted(file_path)

                if is_encrypted:
                    # Создаем временный файл для расшифрованных данных
                    temp_filepath = file_path + '.temp'
                    crypto.decrypt_file(file_path, temp_filepath)

                    with open(temp_filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                    # Удаляем временный файл
                    os.remove(temp_filepath)
                else:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                # Пропускаем файлы, где нет ключа 'is_working_mode'
                if "is_working_mode" not in data or data["is_working_mode"] is None:
                    continue

                # Заменяем 'is_working_mode' на 'mode'
                data["mode"] = data.pop("is_working_mode")

                timestamp_str = data.get("timestamp")
                if timestamp_str is not None:
                    try:
                        dt = datetime.fromisoformat(timestamp_str)
                        timestamp_ms = int(dt.timestamp() * 1000)
                        data["timestamp"] = timestamp_ms
                        total_records += 1
                        logs_data.append(data)
                    except (ValueError, TypeError):
                        continue

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
