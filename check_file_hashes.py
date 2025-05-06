"""
Модуль для проверки хэшей файлов.

Он содержит две функции:
 - calculate_file_hash: вычисляет хэш файла с использованием указанного алгоритма.
 - check_files_hashes: проверяет хэши для списка файлов и выводит результаты.

В примере использования функция check_files_hashes вызывается
с списком файлов, хэши которых нужно проверить.
"""

import hashlib
import os
import json

def calculate_file_hash(file_path, hash_algorithm='sha256'):
    """
    Вычисляет хэш файла с использованием указанного алгоритма.

    :param file_path: путь к файлу
    :param hash_algorithm: алгоритм хэширования (по умолчанию sha256)
    :return: хэш файла
    """
    hash_obj = hashlib.new(hash_algorithm)
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(4096):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        return f"Ошибка: {str(e)}"


def check_files_hashes(file_list):
    """
    Проверяет хэши для списка файлов и выводит результаты.

    :param file_list: список файлов, хэши которых нужно проверить
    """
    print("Проверка хэшей файлов:")
    print("-" * 80)
    print("{:<60} {:<20}".format("Файл", "SHA-256 Хэш"))
    print("-" * 80)

    for file_path in file_list:
        # Убираем дубликаты из списка (если есть)
        if file_list.index(file_path) != file_list.index(file_path):
            continue

        file_path = os.path.normpath(file_path)
        if os.path.exists(file_path):
            file_hash = calculate_file_hash(file_path)
            print("{:<60} {:<20}".format(os.path.relpath(file_path), file_hash))
        else:
            print("{:<60} {:<20}".format(os.path.relpath(file_path), "Файл не найден"))


if __name__ == "__main__":
    # Список файлов, хэши которых нужно проверить
    files_to_check = [
        './device_input/device_log_loader.py',
        './device_input/train_model_mode_by_device_input.py',
        './device_input/predict_mode_by_device_input.py',
        './device_input/listenDeviceInput.py',
        './process_names/train_model_mode_by_processes.py',
        './process_names/processes_log_loader.py',
        './process_names/predict_mode_by_processes.py',
        './process_names/listenProcessesList.py',
    ]

    # загрузка хешей из файла
    try:
        with open('file_hashes.json') as f:
            expected_hashes = json.load(f)
    except FileNotFoundError:
        expected_hashes = {}

    # проверка хэшей
    actual_hashes = {}
    for file_path in files_to_check:
        file_path = os.path.normpath(file_path)
        if os.path.exists(file_path):
            file_hash = calculate_file_hash(file_path)
            actual_hashes[file_path] = file_hash
        else:
            actual_hashes[file_path] = "Файл не найден"

    # сравнение со saved хешами
    for file_path, expected_hash in expected_hashes.items():
        if file_path not in actual_hashes:
            print(f"{file_path}: Файл не найден")
            continue
        if actual_hashes[file_path] != expected_hash:
            print(f"{file_path}: !!! Хеш не совпадает (expected: {expected_hash}, actual: {actual_hashes[file_path]})")
        else:
            print(f"{file_path}: V")

    # сохранение хешей
    with open('file_hashes.json', 'w') as f:
        json.dump(actual_hashes, f, indent=4)

    # Проверка хэшей файлов
    check_files_hashes(files_to_check)