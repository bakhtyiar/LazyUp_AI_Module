import sys
from process import process
from repeat_with_interval import repeat_with_interval
from user_session_manipulator.lock_session import keep_locked

def main():
    print("LazyUp AI Module - Interactive CLI")

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
                input_val = input("Enter monitoring interval in minutes: ")
                if input_val == "b":
                    continue
                input_val = input("Enter lock period in minutes when triggered: ")
                if input_val == "b":
                    continue

                interval = float(input_val)
                lock_period = int(input_val)

                # Convert interval to seconds for repeat_with_interval
                interval_seconds = interval * 60

                # Create monitoring function with specified lock period
                def monitoring_func():
                    process(lock_period)

                # Start monitoring with interval
                stop_event = repeat_with_interval(interval_seconds, monitoring_func)
                print(f"Monitoring started with {interval} minute interval.")
                print("Press Enter to stop monitoring and return to main menu...")
                input()
                stop_event.set()
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
