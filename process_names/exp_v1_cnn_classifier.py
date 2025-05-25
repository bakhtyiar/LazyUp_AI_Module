import time
import tracemalloc
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import optuna
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input, Concatenate, Dropout, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf

from process_names.processes_log_loader import load_processes_logs

# Определение функции focal loss
def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary form of focal loss.
    
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p_t is the probability of the target class.
    
    Parameters:
        gamma: Focusing parameter for modulating loss (default: 2.0)
        alpha: Balancing parameter (default: 0.25)
        
    Returns:
        A callable focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Клипирование для предотвращения деления на ноль
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Вычисление focal loss
        loss = -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred) - \
               (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

# Загрузка данных
data = load_processes_logs(10000)
df = pd.DataFrame(data)
print(df.head())

# Объединяем процессы в одну строку для каждого наблюдения
df["processes_text"] = df["processes"].apply(lambda x: " ".join(x))

# Токенизация процессов
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["processes_text"])
sequences = tokenizer.texts_to_sequences(df["processes_text"])

# Паддинг для одинаковой длины
max_len = max(len(x) for x in sequences)
X_seq = pad_sequences(sequences, maxlen=max_len, padding="post")

# Добавляем timestamp (нормализуем)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_time = scaler.fit_transform(df[["timestamp"]])

# Целевая переменная
y_final = df["mode"].values

# Разделение данных
X_train_seq, X_test_seq, X_train_time, X_test_time, y_train, y_test = train_test_split(
    X_seq, X_time, y_final, test_size=0.2, random_state=42
)

# Функция для отслеживания размеров слоев
class ShapeLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(ShapeLogger, self).__init__()
        self.shapes = {}
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            for layer in self.model.layers:
                if hasattr(layer, 'output_shape'):
                    self.shapes[layer.name] = layer.output_shape
            print("Layer shapes:")
            for name, shape in self.shapes.items():
                print(f"{name}: {shape}")

def create_model(trial):
    # Параметры для Embedding слоя
    embedding_dim = trial.suggest_int('embedding_dim', 16, 128)
    
    # Параметры для Conv1D слоев
    n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
    
    # Входы модели
    input_seq = Input(shape=(max_len,))
    embedding = Embedding(
        input_dim=len(tokenizer.word_index) + 1, 
        output_dim=embedding_dim
    )(input_seq)
    
    # Создание Conv1D слоев
    x = embedding
    current_length = max_len  # Отслеживаем текущую длину
    
    for i in range(n_conv_layers):
        filters = trial.suggest_categorical(f'filters_{i}', [32, 64, 128, 256])
        kernel_size = trial.suggest_int(f'kernel_size_{i}', 2, min(5, current_length))  # Ограничиваем размер ядра
        activation = trial.suggest_categorical(f'activation_conv_{i}', ['relu', 'elu', 'selu'])
        padding = trial.suggest_categorical(f'padding_{i}', ['same', 'valid'])
        
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding=padding
        )(x)
        
        # Обновляем текущую длину после свертки
        if padding == 'valid':
            current_length = current_length - kernel_size + 1
        
        # Добавление MaxPooling
        if trial.suggest_categorical(f'use_pooling_{i}', [True, False]) and current_length > 1:
            # Ограничиваем размер пула
            pool_size = trial.suggest_int(f'pool_size_{i}', 1, min(current_length, 2))
            if pool_size > 1:  # Тільки якщо розмір пулу більше 1
                x = MaxPooling1D(pool_size=pool_size)(x)
                current_length = current_length // pool_size
        
        # Добавление Dropout
        if trial.suggest_categorical(f'use_dropout_conv_{i}', [True, False]):
            dropout_rate = trial.suggest_float(f'dropout_rate_conv_{i}', 0.1, 0.5)
            x = Dropout(dropout_rate)(x)
    
    # Global Pooling (тільки якщо довжина > 1)
    if current_length > 1:
        pooling_type = trial.suggest_categorical('global_pooling', ['max', 'avg'])
        if pooling_type == 'max':
            x = GlobalMaxPooling1D()(x)
        else:
            from tensorflow.keras.layers import GlobalAveragePooling1D
            x = GlobalAveragePooling1D()(x)
    else:
        # Якщо довжина = 1, просто розплюшуємо
        from tensorflow.keras.layers import Flatten
        x = Flatten()(x)
    
    # Вход для timestamp
    input_time = Input(shape=(1,))
    
    # Объединяем
    merged = Concatenate()([x, input_time])
    
    # Полносвязные слои
    n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
    for i in range(n_dense_layers):
        units = trial.suggest_int(f'dense_units_{i}', 16, 256)
        activation = trial.suggest_categorical(f'activation_dense_{i}', ['relu', 'elu', 'selu'])
        merged = Dense(units, activation=activation)(merged)
        
        # Добавление Dropout для Dense слоев
        if trial.suggest_categorical(f'use_dropout_dense_{i}', [True, False]):
            dropout_rate = trial.suggest_float(f'dropout_rate_dense_{i}', 0.1, 0.5)
            merged = Dropout(dropout_rate)(merged)
    
    # Выходной слой
    output = Dense(1, activation='sigmoid')(merged)
    
    # Создание модели
    model = Model(inputs=[input_seq, input_time], outputs=output)
    
    # Оптимизатор
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    if optimizer_name == 'adam':
        beta1 = trial.suggest_float('beta1', 0.8, 0.999)
        beta2 = trial.suggest_float('beta2', 0.8, 0.999)
        optimizer = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    elif optimizer_name == 'rmsprop':
        rho = trial.suggest_float('rho', 0.8, 0.99)
        optimizer = RMSprop(learning_rate=learning_rate, rho=rho)
    else:
        momentum = trial.suggest_float('momentum', 0.0, 0.9)
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    
    # Компиляция модели
    loss_type = trial.suggest_categorical('loss', ['binary_crossentropy', 'binary_focal_loss'])
    if loss_type == 'binary_focal_loss':
        gamma = trial.suggest_float('focal_loss_gamma', 1.0, 3.0)
        alpha = trial.suggest_float('focal_loss_alpha', 0.1, 0.9)
        loss = binary_focal_loss(gamma=gamma, alpha=alpha)
    else:
        loss = 'binary_crossentropy'
        
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def objective(trial):
    # Создание модели с параметрами из trial
    try:
        model = create_model(trial)
        
        # Параметры обучения
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = 50  # Максимальное количество эпох
        
        # Early stopping для предотвращения переобучения
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=trial.suggest_int('patience', 3, 10),
            restore_best_weights=True
        )
        
        # Отслеживание размеров слоев
        shape_logger = ShapeLogger()
        
        # Обучение модели
        history = model.fit(
            [X_train_seq, X_train_time],
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, shape_logger],
            verbose=0
        )
        
        # Оценка на валидационных данных
        val_accuracy = max(history.history['val_accuracy'])
        
        return val_accuracy
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        raise e

# Запуск оптимизации Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # Можно увеличить количество trials для лучших результатов

# Вывод лучших параметров
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Создание лучшей модели
best_model = create_model(trial)

# Измерение использования памяти и времени обучения
tracemalloc.start()
start_train = time.time()

# Обучение лучшей модели на всем тренировочном наборе
best_batch_size = trial.params['batch_size']
patience = trial.params['patience']

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=patience,
    restore_best_weights=True
)

best_model.fit(
    [X_train_seq, X_train_time],
    y_train,
    epochs=50,
    batch_size=best_batch_size,
    validation_split=0.1,
    callbacks=[early_stopping]
)

end_train = time.time()
training_time = end_train - start_train

# Измерение памяти после обучения
current, peak = tracemalloc.get_traced_memory()
max_ram_usage = peak / (1024 ** 2)  # в MB
tracemalloc.stop()

# Оценка качества
start_inf = time.time()
y_pred = best_model.predict([X_test_seq, X_test_time])
end_inf = time.time()
inference_time = end_inf - start_inf

y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
print(classification_report(y_test, y_pred_binary))
print(f"Max RAM Usage: {max_ram_usage:.2f} MB")
print(f"Training time: {training_time:.4f} s")
print(f"Inference time: {inference_time:.4f} s")

# Сохранение лучшей модели
best_model.save('best_cnn_model.h5')