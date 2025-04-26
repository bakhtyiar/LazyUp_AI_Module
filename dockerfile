# Use miniconda as base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

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
    CONDA_DEFAULT_ENV=myenv

# Expose port
EXPOSE 1234

# Set entrypoint
ENTRYPOINT ["conda", "run", "-n", "myenv", "python", "main.py"]