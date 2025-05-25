import process_names.predict_mode_by_processes as predict_process
import device_input.predict_mode_by_device_input as predict_device_input
import user_session_manipulator.lock_session as lock_session

def calculate_true_proportion(lst):
    # Подсчитываем количество True в списке
    true_count = sum(lst)
    # Вычисляем общее количество элементов в списке
    total_count = len(lst)
    # Возвращаем долю True
    return true_count / total_count if total_count > 0 else 0

def compute_status():
    print('predict_device_input.predict()')
    print(predict_device_input.predict_by_device_input())
    process_names_prediction = predict_process.predict_by_processes()
    process_names_prediction = calculate_true_proportion(process_names_prediction)
    device_input_prediction = predict_device_input.predict_by_device_input()
    device_input_prediction = calculate_true_proportion(device_input_prediction)
    if (process_names_prediction == 1 and device_input_prediction == 1):
        return 1
    ret = process_names_prediction / 2 + device_input_prediction / 2
    return ret

def monitor_activity(lock_period: int):
    status = compute_status()
    if status > 0.5:
        lock_session.keep_locked(lock_period)

if __name__ == "__main__":
    compute_status()