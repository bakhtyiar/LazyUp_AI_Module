import os
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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

def setup_plot_controls(fig, ax, timestamps, initial_window_hours=1):
    """Setup zoom controls for the plot"""
    # Add slider for time window adjustment
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    window_slider = Slider(
        ax=slider_ax,
        label='Time Window (hours)',
        valmin=0.5,
        valmax=24,
        valinit=initial_window_hours,
        valstep=0.5
    )
    
    # Add reset zoom button
    reset_ax = plt.axes([0.85, 0.02, 0.1, 0.03])
    reset_button = Button(reset_ax, 'Reset Zoom')
    
    def update_window(val):
        window_size = timedelta(hours=val)
        if timestamps:
            max_time = max(timestamps)
            min_time = max_time - window_size
            ax.set_xlim(min_time, max_time)
            fig.canvas.draw_idle()
    
    def reset_zoom(event):
        if timestamps:
            ax.set_xlim(min(timestamps), max(timestamps))
            window_slider.reset()
            fig.canvas.draw_idle()
    
    window_slider.on_changed(update_window)
    reset_button.on_clicked(reset_zoom)
    
    return window_slider, reset_button

def create_process_predictions_plot():
    """Create interactive plot for process predictions"""
    module_dir = Path(__file__).resolve().parent.parent
    process_pred_dir = os.path.join(module_dir, 'process_names', 'prediction_logs')
    
    data = load_prediction_data(process_pred_dir)
    if not data:
        print("No process prediction data found")
        return
    
    timestamps, predictions = zip(*data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)  # Make room for controls
    
    ax.plot(timestamps, predictions, 'b-', label='Process Predictions')
    ax.set_title('Process-based Mode Predictions Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Prediction Value')
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    
    setup_plot_controls(fig, ax, timestamps)
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
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)  # Make room for controls
    
    ax.plot(timestamps, predictions, 'r-', label='Device Input Predictions')
    ax.set_title('Device Input-based Mode Predictions Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Prediction Value')
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    
    setup_plot_controls(fig, ax, timestamps)
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
    
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)  # Make room for controls
    
    all_timestamps = []
    
    if process_data:
        timestamps, predictions = zip(*process_data)
        all_timestamps.extend(timestamps)
        ax.plot(timestamps, predictions, 'b-', label='Process Predictions', alpha=0.7)
    
    if device_data:
        timestamps, predictions = zip(*device_data)
        all_timestamps.extend(timestamps)
        ax.plot(timestamps, predictions, 'r-', label='Device Input Predictions', alpha=0.7)
    
    ax.set_title('Combined Mode Predictions Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Prediction Value')
    ax.grid(True)
    ax.legend()
    plt.xticks(rotation=45)
    
    if all_timestamps:
        setup_plot_controls(fig, ax, all_timestamps)
    plt.show()

if __name__ == "__main__":
    print("Generating prediction plots...")
    
    # print("\n1. Process Predictions Plot")
    # create_process_predictions_plot()
    # plt.pause(0.1)  # Small pause to ensure window is shown
    #
    # print("\n2. Device Input Predictions Plot")
    # create_device_input_predictions_plot()
    # plt.pause(0.1)  # Small pause to ensure window is shown
    #
    print("\n3. Combined Predictions Plot")
    create_combined_predictions_plot()
    plt.pause(0.1)  # Small pause to ensure window is shown
    
    plt.show(block=True)  # Keep all windows open until manually closed