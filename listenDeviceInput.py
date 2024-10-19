import json
import sys
import signal
from datetime import datetime
import keyboard  # Убедитесь, что эта библиотека установлена

logs = []
is_working_mode = True

def log_event(button_key, is_working_mode):
    timestamp = datetime.now().isoformat()  # Берем текущее время
    log_entry = {
        "timestamp": timestamp,
        "buttonKey": button_key,
        "isWorkingMode": is_working_mode
    }
    logs.append(log_entry)
    with open('device_logs.json', 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=4)

def on_key_event(event):
    global is_working_mode
    try:
        log_event(event.scan_code, is_working_mode)
    except:
        print("Pressed button out from processing range")

def signal_handler(sig, frame):
    print("Прерывание процесса. Сохранение данных и выход.")
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        is_working_mode = sys.argv[1].lower() in ['true', '1', 'yes']

    signal.signal(signal.SIGINT, signal_handler)

    keyboard.hook(on_key_event)  # Слушаем события клавиатуры
    keyboard.wait()  # Ожидаем завершения
