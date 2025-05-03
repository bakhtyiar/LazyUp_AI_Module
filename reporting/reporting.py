import os
import json
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def load_prediction_data(directory):
    """Load prediction data from JSON files in the given directory"""
    data = []
    for file_path in glob.glob(os.path.join(directory, '*.json')):
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            for pred in file_data['predictions']:
                timestamp = datetime.fromisoformat(pred['timestamp'])
                data.append((timestamp, pred['pred']))
    # Sort by timestamp
    data.sort(key=lambda x: x[0])
    return data

def create_process_predictions_plot():
    """Create interactive plot for process predictions"""
    module_dir = Path(__file__).resolve().parent.parent
    process_pred_dir = os.path.join(module_dir, 'process_names', 'prediction_logs')
    
    data = load_prediction_data(process_pred_dir)
    if not data:
        print("No process prediction data found")
        return
    
    timestamps, predictions = zip(*data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, predictions, 'b-', label='Process Predictions')
    plt.title('Process-based Mode Predictions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Prediction Value')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def create_device_input_predictions_plot():
    """Create interactive plot for device input predictions"""
    module_dir = Path(__file__).resolve().parent.parent
    device_pred_dir = os.path.join(module_dir, 'device_input', 'prediction_logs')
    
    data = load_prediction_data(device_pred_dir)
    if not data:
        print("No device input prediction data found")
        return
    
    timestamps, predictions = zip(*data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, predictions, 'r-', label='Device Input Predictions')
    plt.title('Device Input-based Mode Predictions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Prediction Value')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def create_combined_predictions_plot():
    """Create interactive plot combining both prediction types"""
    module_dir = Path(__file__).resolve().parent.parent
    process_pred_dir = os.path.join(module_dir, 'process_names', 'prediction_logs')
    device_pred_dir = os.path.join(module_dir, 'device_input', 'prediction_logs')
    
    process_data = load_prediction_data(process_pred_dir)
    device_data = load_prediction_data(device_pred_dir)
    
    if not process_data and not device_data:
        print("No prediction data found")
        return
    
    plt.figure(figsize=(12, 6))
    
    if process_data:
        timestamps, predictions = zip(*process_data)
        plt.plot(timestamps, predictions, 'b-', label='Process Predictions', alpha=0.7)
    
    if device_data:
        timestamps, predictions = zip(*device_data)
        plt.plot(timestamps, predictions, 'r-', label='Device Input Predictions', alpha=0.7)
    
    plt.title('Combined Mode Predictions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Prediction Value')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Generating prediction plots...")
    
    print("\n1. Process Predictions Plot")
    create_process_predictions_plot()
    
    print("\n2. Device Input Predictions Plot")
    create_device_input_predictions_plot()
    
    print("\n3. Combined Predictions Plot")
    create_combined_predictions_plot()