import os
import json
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from dotenv import load_dotenv

class JsonFolderCrypto:
    def __init__(self, env_file='.env.local'):
        # Загружаем переменные окружения из файла
        load_dotenv(env_file)
        
        # Получаем ключ и соль из переменных окружения
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        self.salt = os.getenv('ENCRYPTION_SALT')
        
        if not self.encryption_key or not self.salt:
            raise ValueError("ENCRYPTION_KEY and ENCRYPTION_SALT must be set in .env.local")
        
        # Преобразуем соль в байты
        salt_bytes = self.salt.encode()
        
        # Создаем ключ для Fernet
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt_file(self, input_path, output_path=None):
        """Шифрует JSON файл"""
        if output_path is None:
            output_path = input_path
        
        # Читаем исходный JSON
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Преобразуем в строку и шифруем
        json_str = json.dumps(data)
        encrypted_data = self.cipher.encrypt(json_str.encode())
        
        # Записываем зашифрованные данные
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
    
    def decrypt_file(self, input_path, output_path=None):
        """Дешифрует JSON файл"""
        if output_path is None:
            output_path = input_path
        
        # Читаем зашифрованные данные
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Дешифруем и преобразуем обратно в JSON
        decrypted_data = self.cipher.decrypt(encrypted_data).decode()
        data = json.loads(decrypted_data)
        
        # Записываем дешифрованные данные
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def encrypt_folder(self, folder_path, output_folder=None):
        """Шифрует все JSON файлы в папке"""
        if output_folder is None:
            output_folder = folder_path
        else:
            os.makedirs(output_folder, exist_ok=True)
        
        for file_path in Path(folder_path).glob('*.json'):
            output_path = Path(output_folder) / file_path.name
            self.encrypt_file(file_path, output_path)
    
    def decrypt_folder(self, folder_path, output_folder=None):
        """Дешифрует все JSON файлы в папке"""
        if output_folder is None:
            output_folder = folder_path
        else:
            os.makedirs(output_folder, exist_ok=True)
        
        for file_path in Path(folder_path).glob('*.json'):
            output_path = Path(output_folder) / file_path.name
            self.decrypt_file(file_path, output_path)

    def is_file_encrypted(self, file_path):
        """
        Проверяет, является ли файл зашифрованным.

        Args:
            file_path: путь к файлу для проверки

        Returns:
            bool: True если файл зашифрован, False если нет
        """
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()

            # Пробуем дешифровать (без сохранения результата)
            self.cipher.decrypt(encrypted_data)
            return True
        except Exception:
            return False

    def is_encrypted_folder(self, folder_path, sample_size=3):
        """
        Проверяет, является ли папка зашифрованной.
        Проверяет несколько случайных файлов из папки (по умолчанию 3).

        Возвращает:
        - True: если все проверенные файлы зашифрованы
        - False: если ни один файл не зашифрован
        - None: если часть файлов зашифрована, а часть нет (неоднозначное состояние)
        """
        json_files = list(Path(folder_path).glob('*.json'))

        if not json_files:
            return False  # Нет файлов для проверки

        # Ограничиваем количество проверяемых файлов
        sample_files = json_files[:sample_size] if len(json_files) > sample_size else json_files

        encrypted_count = 0
        total_checked = 0

        for file_path in sample_files:
            try:
                with open(file_path, 'rb') as f:
                    encrypted_data = f.read()

                # Пробуем дешифровать (без сохранения результата)
                self.cipher.decrypt(encrypted_data)
                encrypted_count += 1
            except Exception:
                # Другие ошибки (например, файл не бинарный)
                pass
            finally:
                total_checked += 1

        if encrypted_count == total_checked and total_checked > 0:
            return True  # Все проверенные файлы зашифрованы
        elif encrypted_count == 0:
            return False  # Ни один файл не зашифрован
        else:
            return None  # Часть файлов зашифрована, часть нет


# Пример использования
if __name__ == "__main__":
    # Инициализируем шифратор
    crypto = JsonFolderCrypto()
    
    print('Encrypting...')
    # Шифруем папку
    crypto.encrypt_folder('device_input/device_input_logs', 'device_input/device_input_logs')
    print('Encrypted')

    x = crypto.is_encrypted_folder('device_input/device_input_logs')
    print('Is encrypted:')
    print(x)

    print('Decrypting...')
    # Дешифруем папку
    crypto.decrypt_folder('device_input/device_input_logs', 'device_input/device_input_logs')
    print('Decrypted')

    x = crypto.is_encrypted_folder('device_input/device_input_logs')
    print('Is encrypted:')
    print(x)
