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

# Install Python 3.10.11 if not installed
if (!(Test-CommandExists python)) {
    Write-Host "Installing Python 3.10.11..." -ForegroundColor Yellow
    choco install python --version=3.10.11 -y
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    $env:Path += ";C:\Python310\Scripts\;C:\Python310\"
    refreshenv
}

# Ensure pip is available
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
if (!(Test-CommandExists pip)) {
    Write-Host "Installing pip..." -ForegroundColor Yellow
    python -m ensurepip --default-pip
    refreshenv
}

# Install pip requirements
Write-Host "Installing pip requirements..." -ForegroundColor Yellow
pip install --upgrade pip

# Check if requirements.txt exists
if (Test-Path -Path "requirements.txt") {
    Write-Host "Installing packages from requirements.txt..." -ForegroundColor Yellow
    try {
        pip install -r requirements.txt
        Write-Host "Successfully installed all packages from requirements.txt" -ForegroundColor Green
    }
    catch {
        Write-Host "Error installing packages. Error: $_" -ForegroundColor Red
        Write-Host "Press any key to exit..."
        $null = $host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
    }
}
else {
    Write-Host "requirements.txt not found in the current directory!" -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
}

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: conda activate $envName" -ForegroundColor Cyan

Write-Host "Press any key to exit..."
$null = $host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')