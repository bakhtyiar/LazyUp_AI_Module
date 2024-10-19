import json
import sys
import signal
from datetime import datetime
import keyboard  # Убедитесь, что эта библиотека установлена
import mouse  # Убедитесь, что эта библиотека установлена
import pprint

# Mouse buttons map
mouse_buttons_map = {
    'left': 1,
    'right': 2,
    'middle': 3,
    'wheel': 4,
    'x': 5,
    'x2': 6,
    'up': 7,
    'down': 8,
    'double': 9,
    'vertical': 10,
    'horizontal': 11,
    'deltaPlus': 12,
    'deltaMinus': 13,
}

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
        

def on_mouse_event(event):
    global is_working_mode
    try:
        if isinstance(event, mouse.ButtonEvent) and event.button in mouse_buttons_map:
            print(event.button)
            print(mouse_buttons_map[event.button])
            log_event(mouse_buttons_map[event.button], is_working_mode)
        elif isinstance(event, mouse.WheelEvent) and event.delta is not None:
            if event.delta > 0:
                log_event(mouse_buttons_map["deltaPlus"], is_working_mode)
            else:
                log_event(mouse_buttons_map["deltaMinus"], is_working_mode)
    except:
        print("Pressed button out from processing range")

def signal_handler(sig, frame):
    print("Прерывание процесса. Сохранение данных и выход.")
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        is_working_mode = sys.argv[1].lower() in ['true', '1', 'yes']

    signal.signal(signal.SIGINT, signal_handler)

    # keyboard.hook(on_key_event)  # Слушаем события клавиатуры
    mouse.hook(on_mouse_event)  # Слушаем события мыши
    # keyboard.wait()  # Ожидаем завершения
    mouse.wait()  # Ожидаем завершения
