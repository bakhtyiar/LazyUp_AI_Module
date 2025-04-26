# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    x11-xserver-utils \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml .

# Create conda environment with explicit platform-independent channels
RUN conda config --set channel_priority strict \
    && conda env create -f environment.yml \
    && conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Copy application files
COPY . .

# Create directories for persistent data if they don't exist
RUN mkdir -p device_input process_names gui

# Set environment variables for better cross-platform compatibility
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISPLAY=:99 \
    CONDA_DEFAULT_ENV=myenv

# Expose port
EXPOSE 1234

# Create entrypoint script for proper environment setup
RUN echo '#!/bin/bash\nxvfb-run -a conda run -n myenv python main.py' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]