import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

# Функция для запуска/остановки скрипта
def toggle_recording():
    global recording_process
    if recording_process is None:
        # Запуск скрипта listenProcessesList.py и listenDeviceInput.py
        recording_process = subprocess.Popen([sys.executable, module_dir + "/../process_names/listenProcessesList.py", current_arg])
        recording_process = subprocess.Popen([sys.executable, module_dir + "/../device_input/listenDeviceInput.py", current_arg])
        record_button.config(text="Остановить запись", style="TButton")
        status_label.config(text="Запись запущена", foreground="green")
    else:
        # Остановка скрипта
        recording_process.terminate()
        recording_process = None
        record_button.config(text="Запустить запись", style="TButton")
        status_label.config(text="Запись остановлена", foreground="grey")

# Функции для установки sys.argv[1]
def set_working():
    global current_arg
    current_arg = "true"
    update_script_arg()
    alpha_button.config(style="Selected.TButton")
    beta_button.config(style="TButton")

def set_relaxing():
    global current_arg
    current_arg = "false"
    update_script_arg()
    beta_button.config(style="Selected.TButton")
    alpha_button.config(style="TButton")

# Функция для обновления аргумента скрипта
def update_script_arg():
    global recording_process, current_arg
    if recording_process is not None:
        recording_process.terminate()
        recording_process = subprocess.Popen([sys.executable, module_dir + "/process_names/listenProcessesList.py", current_arg])
        recording_process = subprocess.Popen([sys.executable, module_dir + "/device_input/listenDeviceInput.py", current_arg])

# Инициализация главного окна
root = tk.Tk()
root.title("Управление записью")
root.geometry("300x200")  # Установка размера окна
root.resizable(False, False)  # Запрет изменения размера окна

# Стили для кнопок
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("Selected.TButton", font=("Helvetica", 12, "bold"), padding=10, background="lightblue")

# Переменные для управления процессом и аргументом
recording_process = None
current_arg = "true"

# Создание фрейма для кнопок
button_frame = ttk.Frame(root)
button_frame.pack(pady=20)

# Кнопка "Запустить/Остановить запись"
record_button = ttk.Button(button_frame, text="Запустить запись", command=toggle_recording)
record_button.pack(pady=10)

# Кнопки "Труд" и "Отдых"
alpha_button = ttk.Button(button_frame, text="Труд", command=set_working, style="Selected.TButton")
alpha_button.pack(side=tk.LEFT, padx=10)

beta_button = ttk.Button(button_frame, text="Отдых", command=set_relaxing, style="TButton")
beta_button.pack(side=tk.RIGHT, padx=10)

# Метка для отображения статуса
status_label = ttk.Label(root, text="Запись остановлена", font=("Helvetica", 12), foreground="grey")
status_label.pack(pady=10)

# Запуск главного цикла обработки событий
root.mainloop()
