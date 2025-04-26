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
    return stopped  # Возвращаем только Event для управления

def example():
    stopper = repeat_with_interval(1, lambda: print("Tick"))
    time.sleep(10)
    stopper.set()  # Останавливаем выполнение
    # Поток завершится автоматически, так как stopper.set() прервет wait()

if __name__ == "__main__":
    example()