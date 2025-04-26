import time
import threading

def repeat_with_interval(interval, func):
    stopped = threading.Event()

    def loop():
        while not stopped.wait(interval):
            func()

    t = threading.Thread(target=loop)
    t.daemon = True  # Добавим демонизацию потока для автоматического завершения при выходе
    t.start()
    return stopped

def repeat_with_interval_patient(interval, func):
    stopped = threading.Event()
    func_lock = threading.Lock()
    func_running = False

    def loop():
        nonlocal func_running
        while not stopped.wait(interval):
            with func_lock:
                if func_running:
                    # Если предыдущий вызов еще выполняется, ждем его завершения
                    continue
                func_running = True

            try:
                func()
            finally:
                with func_lock:
                    func_running = False

    t = threading.Thread(target=loop)
    t.daemon = True  # Добавим демонизацию потока для автоматического завершения при выходе
    t.start()
    return stopped

def example():
    stopper = repeat_with_interval(1, lambda: print("Tick"))
    time.sleep(10)
    stopper.set()  # Останавливаем выполнение
    # Поток завершится автоматически, так как stopper.set() прервет wait()

if __name__ == "__main__":
    example()