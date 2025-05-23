import pyuac
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna
import autokeras as ak
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from pathlib import Path
import pickle

# Import the data loader
from device_log_loader import load_device_logs

# Add patch for NumPy object deprecation in AutoKeras
# This addresses the error: AttributeError: module 'numpy' has no attribute 'object'.
import numpy
if not hasattr(numpy, 'object'):
    numpy.object = object  # Patch for newer NumPy versions

class DeviceLogCNNClassifier:
    def __init__(self, 
                 n_splits=5, 
                 optuna_trials=50,
                 autokeras_trials=10,
                 max_files=None, 
                 max_units=None,
                 model_save_path='models/device_cnn_model.h5',
                 scaler_save_path='models/device_scaler.pkl',
                 random_state=42):
        """
        Initialize the CNN classifier for device log data.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            optuna_trials: Number of trials for Optuna hyperparameter optimization
            autokeras_trials: Number of trials for AutoKeras architecture search
            max_files: Maximum number of files to load
            max_units: Maximum number of units to load
            model_save_path: Path to save the trained model
            scaler_save_path: Path to save the feature scaler
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.optuna_trials = optuna_trials
        self.autokeras_trials = autokeras_trials
        self.max_files = max_files
        self.max_units = max_units
        self.model_save_path = model_save_path
        self.scaler_save_path = scaler_save_path
        self.random_state = random_state
        
        # Initialize model and scaler as None
        self.model = None
        self.scaler = None
        
        # Create directories for model saving if they don't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    def load_and_preprocess_data(self):
        """
        Load device log data and preprocess it for CNN training.
        
        Returns:
            X: Feature matrix
            y: Target vector
        """
        # Load data using the device_log_loader
        data = load_device_logs(max_files=self.max_files, max_units=self.max_units)
        
        if not data:
            raise ValueError('No data loaded. Check device_input_logs directory.')
        
        # Prepare lists for features and targets
        X_list = []
        y_list = []
        
        # Process each log file
        for log_entry in data:
            mode = log_entry['mode']  # Target value (0 or 1)
            
            # Process each button press in the log
            for item in log_entry['list']:
                button_key = item['buttonKey']
                timestamp = item['dateTime']
                
                # Extract time features from timestamp
                dt = datetime.fromtimestamp(timestamp / 1000)
                hour = dt.hour
                minute = dt.minute
                second = dt.second
                day_of_week = dt.weekday()
                
                # Create feature vector
                features = [button_key, hour, minute, second, day_of_week]
                X_list.append(features)
                y_list.append(mode)
        
        # Convert lists to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Save scaler for future use
        os.makedirs(os.path.dirname(self.scaler_save_path), exist_ok=True)
        with open(self.scaler_save_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return X_scaled, y
    
    def build_cnn_model(self, trial=None):
        """
        Build a CNN model for device log data classification.
        
        Args:
            trial: Optuna trial object for hyperparameter optimization
            
        Returns:
            Compiled CNN model
        """
        if trial:
            # Define hyperparameters to optimize with Optuna
            filters1 = trial.suggest_categorical('filters1', [16, 32, 64])
            filters2 = trial.suggest_categorical('filters2', [32, 64, 128])
            kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
            dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            
            # Activation functions to optimize
            conv_activation = trial.suggest_categorical('conv_activation', ['relu', 'elu', 'selu', 'tanh', 'swish'])
            dense_activation = trial.suggest_categorical('dense_activation', ['relu', 'elu', 'selu', 'tanh', 'swish'])
            final_activation = trial.suggest_categorical('final_activation', ['sigmoid', 'softmax'])
        else:
            # Default hyperparameters
            filters1 = 32
            filters2 = 64
            kernel_size = 3
            dense_units = 128
            dropout_rate = 0.3
            learning_rate = 0.001
            
            # Default activation functions
            conv_activation = 'relu'
            dense_activation = 'relu'
            final_activation = 'sigmoid'
        
        # Get input shape from data
        X, _ = self.load_and_preprocess_data()
        input_shape = (X.shape[1], 1)  # (n_features, 1) for 1D CNN
        
        # Build model
        model = models.Sequential()
        
        # First convolutional layer
        model.add(layers.Conv1D(filters=filters1, kernel_size=kernel_size, activation=conv_activation, 
                               input_shape=input_shape, padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Second convolutional layer
        model.add(layers.Conv1D(filters=filters2, kernel_size=kernel_size, activation=conv_activation, padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Flatten layer
        model.add(layers.Flatten())
        
        # Dense layers
        model.add(layers.Dense(dense_units, activation=dense_activation))
        model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation=final_activation))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Mean validation accuracy across time series splits
        """
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        accuracies = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Reshape for CNN (samples, features, channels)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            # Build model with trial hyperparameters
            model = self.build_cnn_model(trial)
            
            # Early stopping callback
            # todo: вернуть patience 5
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            # Train model
            # todoL вернуть epochs 30
            model.fit(X_train, y_train, 
                     epochs=10,  
                     batch_size=32, 
                     validation_split=0.2,
                     callbacks=[early_stopping],
                     verbose=0)
            
            # Evaluate model
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        # Return mean accuracy across splits
        return np.mean(accuracies)
    
    def optimize_hyperparameters(self):
        """
        Optimize CNN hyperparameters using Optuna.
        
        Returns:
            Dictionary of best hyperparameters
        """
        print('Starting Optuna hyperparameter optimization...')
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.optuna_trials)
        
        # Get best hyperparameters
        best_params = study.best_params
        print(f'Best hyperparameters: {best_params}')
        print(f'Best accuracy: {study.best_value:.4f}')
        
        # Visualize optimization results
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image('models/optuna_history.png')
            
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image('models/optuna_param_importances.png')
        except:
            print('Could not generate Optuna visualization.')
        
        return best_params
    
    def train_with_best_params(self, best_params):
        """
        Train CNN model with best hyperparameters from Optuna.
        
        Args:
            best_params: Dictionary of best hyperparameters
            
        Returns:
            Trained CNN model
        """
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Reshape for CNN (samples, features, channels)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Create model with best hyperparameters
        model = models.Sequential()
        
        # First convolutional layer
        model.add(layers.Conv1D(filters=best_params['filters1'], 
                              kernel_size=best_params['kernel_size'], 
                              activation=best_params['conv_activation'], 
                              input_shape=(X_train.shape[1], 1), 
                              padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Second convolutional layer
        model.add(layers.Conv1D(filters=best_params['filters2'], 
                              kernel_size=best_params['kernel_size'], 
                              activation=best_params['conv_activation'], 
                              padding='same'))
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Flatten layer
        model.add(layers.Flatten())
        
        # Dense layers
        model.add(layers.Dense(best_params['dense_units'], activation=best_params['dense_activation']))
        model.add(layers.Dropout(best_params['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation=best_params['final_activation']))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=best_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks
        # todo : вернуть patience 10
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(self.model_save_path, save_best_only=True, monitor='val_loss')
        
        # Train model
        # вернуть epochs 100
        history = model.fit(
            X_train, y_train,
            epochs=20,  
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Evaluate model
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Test accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('models/confusion_matrix.png')
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        
        return model
    
    def search_architecture(self):
        """
        Search for optimal neural network architecture using AutoKeras.
        
        Returns:
            Best AutoKeras model
        """
        print('Starting AutoKeras neural architecture search...')
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Create a specific directory for AutoKeras in the user's home directory to avoid permission issues
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="autokeras_")
        print(f"Using temporary directory for AutoKeras: {temp_dir}")
        
        # Initialize AutoKeras structured data classifier with the temporary directory
        clf = ak.StructuredDataClassifier(
            overwrite=True,
            max_trials=self.autokeras_trials,
            project_name=temp_dir
        )
        
        # Search for best architecture
        clf.fit(X_train, y_train, validation_split=0.2, epochs=10)  
        
        # Evaluate best model
        y_pred = (clf.predict(X_test) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'AutoKeras test accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))
        
        # Get the best model
        model = clf.export_model()
        
        return model
    
    def fit(self, use_optuna=True, use_autokeras=True):
        """
        Train the CNN classifier using Optuna for hyperparameter optimization and/or
        AutoKeras for architecture search.
        
        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization
            use_autokeras: Whether to use AutoKeras for architecture search
            
        Returns:
            self
        """
        if use_optuna:
            # Optimize hyperparameters with Optuna
            best_params = self.optimize_hyperparameters()
            
            # Train model with best hyperparameters
            self.model = self.train_with_best_params(best_params)
        
        if use_autokeras:
            # Search for optimal architecture with AutoKeras
            autokeras_model = self.search_architecture()
            
            # Save AutoKeras model
            autokeras_model.save('models/autokeras_model')
            
            # Compare with Optuna model if both were used
            if use_optuna and self.model:
                # Load and preprocess data
                X, y = self.load_and_preprocess_data()
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state, stratify=y
                )
                
                # Reshape for CNN
                X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                # Get predictions from both models
                y_pred_optuna = (self.model.predict(X_test_cnn) > 0.5).astype(int)
                y_pred_autokeras = (autokeras_model.predict(X_test) > 0.5).astype(int)
                
                # Compare accuracies
                acc_optuna = accuracy_score(y_test, y_pred_optuna)
                acc_autokeras = accuracy_score(y_test, y_pred_autokeras)
                
                print(f'Optuna model accuracy: {acc_optuna:.4f}')
                print(f'AutoKeras model accuracy: {acc_autokeras:.4f}')
                
                # Use the better model
                if acc_autokeras > acc_optuna:
                    print('AutoKeras model performs better. Using it as the final model.')
                    self.model = autokeras_model
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained CNN model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        if self.model is None:
            raise ValueError('Model not trained. Call fit() first.')
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Reshape for CNN
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Make predictions
        y_pred = (self.model.predict(X_reshaped) > 0.5).astype(int)
        
        return y_pred
    
    def save_model(self, path=None):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model to
        """
        if self.model is None:
            raise ValueError('Model not trained. Call fit() first.')
        
        save_path = path if path else self.model_save_path
        self.model.save(save_path)
        print(f'Model saved to {save_path}')
    
    def load_model(self, path=None):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        load_path = path if path else self.model_save_path
        self.model = tf.keras.models.load_model(load_path)
        print(f'Model loaded from {load_path}')

# Example usage
if __name__ == '__main__':
    if not pyuac.isUserAdmin():
        print("Re-launching as admin!")
        pyuac.runAsAdmin()

    # Create CNN classifier
    classifier = DeviceLogCNNClassifier(
        # todo: вернуть настройки
        n_splits=2,  # 5
        optuna_trials=3, # 20
        autokeras_trials=2,  # 5
        max_files=None,  
        max_units=None,  
        model_save_path='models/device_cnn_model.h5',
        scaler_save_path='models/device_scaler.pkl',
        random_state=42
    )
    
    # Train the model with both Optuna and AutoKeras
    classifier.fit(use_optuna=True, use_autokeras=True)
    
    # Save the trained model
    classifier.save_model()
    
    # Example of loading the model and making predictions
    print('\nExample of loading the model and making predictions:')
    new_classifier = DeviceLogCNNClassifier()
    new_classifier.load_model()
    
    # Load some test data
    X, y = new_classifier.load_and_preprocess_data()
    X_test = X[:5]  
    
    # Make predictions
    predictions = new_classifier.predict(X_test)
    print(f'Predictions for 5 test samples: {predictions}')