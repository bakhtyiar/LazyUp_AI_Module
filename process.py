import process_names.predict_mode_by_processes as predict_process
import user_session_manipulator.lock_session as lock_session

def main():
    ret = predict_process.predict()
    if ret[0][0] > 0.5:
        lock_session.lock_windows()
    print (ret)

if __name__ == "__main__":
    main()