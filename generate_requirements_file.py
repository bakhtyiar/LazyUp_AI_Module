import subprocess
import os


def generate_requirements(project_path=".", output_file="requirements.txt"):
    try:
        project_path = os.path.abspath(project_path)
        requirements_file = os.path.join(project_path, output_file)

        # Запускаем pipreqs для указанной директории
        subprocess.run(["pipreqs", project_path, "--force"], check=True)

        # Переименовываем созданный файл
        if project_path != ".":
            os.rename(os.path.join(project_path, "requirements.txt"), requirements_file)

        print(f"Файл {requirements_file} успешно создан!")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    project_path = input("Введите путь к проекту (по умолчанию текущая папка): ").strip() or "."
    generate_requirements(project_path)