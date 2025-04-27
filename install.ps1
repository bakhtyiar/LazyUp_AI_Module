Write-Host "Setting up LazyUp AI Module environment..." -ForegroundColor Green

# Check if Python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed. Please install Python first." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (!(Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
