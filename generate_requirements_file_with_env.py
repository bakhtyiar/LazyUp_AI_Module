import subprocess
import os

def generate_requirements(project_path=".", output_file="requirements.txt"):
    try:
        project_path = os.path.abspath(project_path)
        requirements_file = os.path.join(project_path, output_file)

        # Запускаем pip freeze и записываем вывод в файл
        with open(output_file, "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, check=True)

        print(f"Файл {output_file} успешно создан!")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    project_path = input("Введите путь к проекту (по умолчанию текущая папка): ").strip() or "."
    generate_requirements(project_path)