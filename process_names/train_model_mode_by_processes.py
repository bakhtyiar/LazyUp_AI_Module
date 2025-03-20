import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM
import json
import os
import process_name_tokenizing.process_name_tokens_manager as process_name_tokenizing

module_dir = os.path.dirname(os.path.abspath(__file__))

log_data = []
log_directory = module_dir + './processes_logs'

# Загрузка данных
for filename in os.listdir(log_directory):
    if filename.endswith('.json'):  # Проверяем, что файл имеет расширение .json
        file_path = os.path.join(log_directory, filename)  # Полный путь к файлу
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)  # Загружаем данные из JSON-файла
                # Проверяем структуру JSON
                if (
                    isinstance(data, dict) and
                    'is_working_mode' in data and
                    'timestamp' in data and
                    'processes' in data and
                    isinstance(data['processes'], list)
                ):
                    log_data.append(data)  # Добавляем данные в список
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при обработке файла {filename}: {e}")
data = log_data
df = pd.DataFrame(data)

# Подготовка
max_length = 64  # Максимальная длина последовательности

tokenizer = process_name_tokenizing.process_tokenization(df['processes'])

X = tokenizer.texts_to_sequences(df['processes'])
X = pad_sequences(X, maxlen=max_length)
y = pd.get_dummies(df['is_working_mode']).values  # one-hot encoding

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Модель
model = None
try:
    model = load_model(module_dir + './predict_processes.h5')
except OSError:
    print("Saved model not found")
if not model:
    model = Sequential()
    amount_of_different_words = (len(tokenizer.word_index) + 1)
    model.add(
        Embedding(input_dim=amount_of_different_words, output_dim=amount_of_different_words, input_length=max_length))
    model.add(LSTM(amount_of_different_words))
    model.add(Dense(2, activation='softmax'))
    # Компиляция
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Обучение
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    model.save(module_dir + './predict_processes.h5')

# Оценка
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
