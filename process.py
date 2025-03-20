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

def main():
    print('predict_device_input.predict()')
    print(predict_device_input.predict())
    process_names_prediction = predict_process.predict()[0][0]
    device_input_prediction = calculate_true_proportion(predict_device_input.predict())
    ret = process_names_prediction / 2 + device_input_prediction / 2
    return ret

if __name__ == "__main__":
    main()