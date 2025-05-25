import os
import sys

from device_input.listenDeviceInput import start_listening as start_device_listening, \
    stop_listening as stop_device_listening
from monitor_activity import monitor_activity
from process_names.listenProcessesList import init_process_logging
from repeat_with_interval import repeat_with_interval
from user_session_manipulator.lock_session import keep_locked


def show_agreement_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            print("\n" + "=" * 80)
            print(content)
            print("=" * 80)
            print("\nДля согласия введите Y , для отказа - любую другую клавишу")
            response = input()
            return response == "Y" or response == "y"
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден")
        return False


def check_first_run():
    flag_file = ".agreement_complete"
    if not os.path.exists(flag_file):
        print("Первый запуск приложения. Необходимо ознакомиться с документами...")

        # Show Privacy Policy
        if not show_agreement_file("PRIVACY_POLICY_ru.md"):
            print("Вы не приняли политику конфиденциальности. Приложение будет закрыто.")
            sys.exit(1)

        # Show Instructions
        if not show_agreement_file("README_instruction_ru.md"):
            print("Вы не приняли инструкцию. Приложение будет закрыто.")
            sys.exit(1)

        # Create flag file to mark successful agreement
        with open(flag_file, 'w') as f:
            f.write("1")
        print("\nСпасибо за принятие условий! Запуск приложения...")


def main():
    print("LazyUp AI Module - Interactive CLI")

    # Check for first run
    check_first_run()

    while True:
        try:
            print("\nCommands:")
            print("1. Lock session")
            print("2. Start monitoring with interval")
            print("3. Exit")

            choice = input("\nEnter your choice (1-3): ")

            if choice == "1":
                print("Input 'b' to go back.")
                input_val = input("Enter lock duration in minutes: ")
                if input_val == "b":
                    continue
                duration = int(input_val)
                keep_locked(duration)

            elif choice == "2":
                print("Input 'b' to go back.")
                input_valA = input("Enter checking activity interval in minutes: ")
                if input_valA == "b":
                    continue
                input_valB = input("Enter lock period in minutes when triggered: ")
                if input_valB == "b":
                    continue

                interval = float(input_valA)
                lock_period = int(input_valB)

                # Convert interval to seconds for repeat_with_interval
                interval_seconds = interval * 60

                # Create monitoring function with specified lock period
                def monitoring_func():
                    monitor_activity(lock_period)

                # Start monitoring with interval
                stop_event = repeat_with_interval(interval_seconds, monitoring_func)

                # Start device input listening
                start_device_listening(working_mode=None)

                # Start process monitoring (default 5 second interval)
                process_logging_timer = init_process_logging(
                    is_working_mode=None,
                    logs_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'process_names',
                                          'processes_logs'),
                    time_interval=5
                )

                print(f"Monitoring started with {interval} minute interval.")
                print("Press Enter to stop monitoring and return to main menu...")
                input()

                # Stop all monitoring
                stop_event.set()
                stop_device_listening()
                process_logging_timer.cancel()

                print("Monitoring stopped.")

            elif choice == "3":
                print("Exiting...")
                sys.exit(0)

            else:
                print("Invalid choice. Please enter a number between 1 and 3.")

        except ValueError as e:
            print("Invalid input.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    main()
