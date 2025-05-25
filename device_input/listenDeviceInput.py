import json
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

import keyboard  # Убедитесь, что эта библиотека установлена
import mouse  # Убедитесь, что эта библиотека установлена

from logs_cypher import JsonFolderCrypto

# Mouse buttons map
mouse_buttons_map = {
    'left': 151,
    'right': 152,
    'middle': 153,
    'wheel': 154,
    'x': 155,
    'x2': 156,
    'up': 157,
    'down': 158,
    'double': 159,
    'vertical': 160,
    'horizontal': 161,
    'deltaPlus': 162,
    'deltaMinus': 163,
}

logs = []
is_working_mode = True
current_log_file = None
module_dir = Path(__file__).resolve().parent
directory_path = os.path.join(module_dir, 'device_input_logs')  # Путь к директории с JSON-файлами
crypto = JsonFolderCrypto()

# Глобальные переменные для отслеживания хуков
keyboard_hook = None
mouse_hook = None
is_listening = False


def get_log_filename():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"


def init_log_file(directory=directory_path):
    global current_log_file, logs
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    current_log_file = os.path.join(directory, get_log_filename())
    logs = []  # Очищаем логи при создании нового файла


def log_event(button_key, is_working_mode=None):
    global current_log_file, logs
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "buttonKey": button_key,
        "isWorkingMode": is_working_mode
    }
    if is_working_mode is not None:
        log_entry['isWorkingMode'] = is_working_mode
    logs.append(log_entry)
    with open(current_log_file, 'w', encoding='utf-8') as f:
        json.dump({"deviceLogs": logs}, f, indent=4)
    # Шифруем файл после записи
    crypto.encrypt_file(current_log_file)


def on_key_event(event):
    global is_working_mode
    try:
        print(event.name, event.scan_code)
        log_event(event.scan_code, is_working_mode)
    except:
        print("Pressed button out from processing range")


def on_mouse_event(event):
    global is_working_mode
    try:
        if isinstance(event, mouse.ButtonEvent) and event.button in mouse_buttons_map:
            print(event.button, mouse_buttons_map[event.button])
            log_event(mouse_buttons_map[event.button], is_working_mode)
        elif isinstance(event, mouse.WheelEvent) and event.delta is not None:
            if event.delta > 0:
                log_event(mouse_buttons_map["deltaPlus"], is_working_mode)
            else:
                log_event(mouse_buttons_map["deltaMinus"], is_working_mode)
    except:
        print("Pressed button out from processing range")


def start_listening(working_mode=None, directory=directory_path):
    """Start listening to keyboard and mouse events"""
    global is_working_mode, is_listening, keyboard_hook, mouse_hook

    if is_listening:
        print("Already listening to events")
        return

    is_working_mode = working_mode
    init_log_file(directory)  # Инициализируем файл логов при старте

    # Устанавливаем хуки для клавиатуры и мыши
    keyboard_hook = keyboard.hook(on_key_event)
    mouse_hook = mouse.hook(on_mouse_event)
    is_listening = True
    print("Started listening to device inputs")


def stop_listening():
    """Stop listening to keyboard and mouse events"""
    global is_listening, keyboard_hook, mouse_hook

    if not is_listening:
        print("Not currently listening to events")
        return

    # Удаляем хуки
    keyboard.unhook(keyboard_hook)
    mouse.unhook(mouse_hook)
    is_listening = False
    print("Stopped listening to device inputs")


def signal_handler(sig, frame):
    stop_listening()
    print("Прерывание процесса. Сохранение данных и выход.")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        is_working_mode = sys.argv[1].lower() in ['true', '1', 'yes']

    signal.signal(signal.SIGINT, signal_handler)
    start_listening(is_working_mode)

    try:
        keyboard.wait()  # Ожидаем завершения
    except KeyboardInterrupt:
        stop_listening()
        print("Прерывание процесса. Сохранение данных и выход.")
