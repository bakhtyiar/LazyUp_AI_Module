import ctypes
import time
import psutil  # Для проверки состояния системы
from datetime import datetime, timedelta

def lock_windows():
    # Блокирует компьютер.
    ctypes.windll.user32.LockWorkStation()

def is_locked():
    # Проверяет, заблокирован ли компьютер.
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'LogonUI.exe':  # LogonUI.exe появляется при блокировке
            return True
    return False

def keep_locked(lock_duration_minutes):
    # Блокирует компьютер и следит за его состоянием в течение указанного времени.
    # :param lock_duration_minutes: Время блокировки в минутах.
    lock_windows()  # Блокируем компьютер
    end_time = datetime.now() + timedelta(minutes=lock_duration_minutes)  # Время окончания блокировки

    while datetime.now() < end_time:  # Пока не истекло время блокировки
        time.sleep(1)  # Проверяем состояние каждую секунду
        if not is_locked():  # Если компьютер разблокирован
            print("Компьютер разблокирован. Блокируем снова...")
            lock_windows()  # Снова блокируем

    print("Время блокировки истекло. Скрипт завершает работу.")

# Запускаем блокировку на 1 минуту
keep_locked(1)