import json
import sys
import signal
from datetime import datetime
import keyboard  # Убедитесь, что эта библиотека установлена
import mouse  # Убедитесь, что эта библиотека установлена

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

def signal_handler(sig, frame):
    print("Прерывание процесса. Сохранение данных и выход.")
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        is_working_mode = sys.argv[1].lower() in ['true', '1', 'yes']

    signal.signal(signal.SIGINT, signal_handler)

    keyboard.hook(on_key_event)  # Слушаем события клавиатуры
    mouse.hook(on_mouse_event)  # Слушаем события мыши
    keyboard.wait()  # Ожидаем завершения
    mouse.wait()  # Ожидаем завершения
