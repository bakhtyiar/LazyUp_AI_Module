import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import subprocess

def update_config(minutes):
    config_path = 'config.json'
    config = {'lock_duration': minutes}
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

def validate_minutes():
    try:
        minutes = int(minutes_entry.get())
        if minutes < 1:
            minutes_entry.delete(0, tk.END)
            minutes_entry.insert(0, "1")
            lock_button.config(state='normal')
            return False
        else:
            lock_button.config(state='normal')
            return True
    except ValueError:
        minutes_entry.delete(0, tk.END)
        minutes_entry.insert(0, "1")
        lock_button.config(state='normal')
        return False

def start_lock():
    if validate_minutes():
        try:
            minutes = int(minutes_entry.get())
            update_config(minutes)
            import user_session_manipulator.lock_session as lock_session
            lock_session.keep_locked()
            minutes_entry.config(state='disabled')
            lock_button.config(state='disabled')
            messagebox.showinfo("Успех", "Блокировка запущена!")
        except ImportError:
            messagebox.showerror("Ошибка", "Модуль lock_session не найден!")

def start_monitoring():
    try:
        subprocess.Popen(['python', 'process.py'])
        messagebox.showinfo("Успех", "Мониторинг прокрастинации запущен!")
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл process.py не найден!")

root = tk.Tk()
root.title("Управление блокировкой и мониторингом")
root.geometry("350x220")  # Увеличим высоту окна для лучшего размещения элементов
root.resizable(True, True)

style = ttk.Style()
style.configure("TButton", padding=10, font=("Helvetica", 12))
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 12))

# Создаем фрейм для лучшей организации элементов
main_frame = ttk.Frame(root)
main_frame.pack(padx=20, pady=20, fill='both', expand=True)

# Поле ввода минут
minutes_label = ttk.Label(main_frame, text="Минуты:")
minutes_label.pack(pady=(0,5))

minutes_entry = ttk.Entry(main_frame, width=10)
minutes_entry.pack(pady=(0,15))
minutes_entry.insert(0, "1")  # Начальное значение (минимум 1)
minutes_entry.bind("<FocusOut>", lambda event: validate_minutes())

# Кнопка запуска блокировки
lock_button = ttk.Button(main_frame, text="Запустить блокировку", command=start_lock)
lock_button.pack(pady=(0,5))

# Кнопка запуска мониторинга
monitor_button = ttk.Button(main_frame, text="Запустить мониторинг прокрастинации", command=start_monitoring)
monitor_button.pack(pady=(0,5))

# Валидация минут при запуске
validate_minutes()

root.mainloop()