# Script to set up LazyUp AI Module environment
Write-Host "Setting up LazyUp AI Module environment..." -ForegroundColor Green

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { return $true } }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Install Chocolatey if not installed
if (!(Test-CommandExists choco)) {
    Write-Host "Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    refreshenv
}

# Install Python 3.10.13 if not installed
if (!(Test-CommandExists python)) {
    Write-Host "Installing Python 3.10.13..." -ForegroundColor Yellow
    choco install python --version=3.10.13 -y
    refreshenv
}

# Install Miniconda if not installed
if (!(Test-CommandExists conda)) {
    Write-Host "Installing Miniconda..." -ForegroundColor Yellow
    choco install miniconda3 -y
    refreshenv
}

# Initialize conda for PowerShell
conda init powershell

# Create and activate conda environment
$envName = "myenv"
Write-Host "Creating conda environment '$envName'..." -ForegroundColor Yellow

# Remove existing environment if it exists
conda env remove --name $envName -y

# Create new environment with Python 3.10.13
conda create -n $envName python=3.10.13 -y

# Activate conda environment
Write-Host "Activating conda environment..." -ForegroundColor Yellow
conda activate $envName

# Install pip requirements
Write-Host "Installing pip requirements..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: conda activate $envName" -ForegroundColor Cyan
