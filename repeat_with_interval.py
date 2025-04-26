import time
import threading
import logging
from typing import Callable

def repeat_with_interval(interval: float, func: Callable[[], None]) -> threading.Event:
    """
    Запускает функцию в отдельном потоке с заданным интервалом.

    Args:
        interval: Интервал в секундах между вызовами функции
        func: Функция, которую нужно вызывать периодически

    Returns:
        Объект threading.Event, установка которого остановит выполнение
    """
    stopped = threading.Event()

    def loop():
        while not stopped.wait(interval):
            func()

    t = threading.Thread(target=loop)
    t.daemon = True  # Добавим демонизацию потока для автоматического завершения при выходе
    t.start()
    return stopped


def repeat_with_interval_patient(interval: float, func: Callable[[], None]) -> threading.Event:
    """
    Запускает функцию в отдельном потоке с заданным интервалом.

    Args:
        interval: Интервал в секундах между вызовами функции
        func: Функция, которую нужно вызывать периодически

    Returns:
        Объект threading.Event, установка которого остановит выполнение
    """
    stopped = threading.Event()
    func_lock = threading.Lock()
    func_running = False

    def loop() -> None:
        nonlocal func_running
        while not stopped.is_set():
            # Ждем указанный интервал или пока не будет останов
            if stopped.wait(interval):
                break

            with func_lock:
                if func_running:
                    # Если предыдущий вызов еще выполняется, ждем его завершения
                    continue
                func_running = True

            try:
                func()
            except Exception as e:
                logging.exception(f"Error in repeated function: {e}")
            finally:
                with func_lock:
                    func_running = False

    t = threading.Thread(target=loop, name="IntervalThread")
    t.daemon = True  # Добавим демонизацию потока для автоматического завершения при выходе
    t.start()
    return stopped
def example():
    stopper = repeat_with_interval_patient(1, lambda: print("Tick"))
    time.sleep(10)
    stopper.set()  # Останавливаем выполнение
    # Поток завершится автоматически, так как stopper.set() прервет wait()

if __name__ == "__main__":
    example()