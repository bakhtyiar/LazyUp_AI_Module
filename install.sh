#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
CYAN='\033[0;36m'

echo -e "${GREEN}Setting up LazyUp AI Module environment...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect package manager
get_package_manager() {
    if command_exists apt-get; then
        echo "apt"
    elif command_exists dnf; then
        echo "dnf"
    else
        echo "unknown"
    fi
}

# Install Python if not installed
if ! command_exists python3.10; then
    echo -e "${YELLOW}Installing Python 3.10...${NC}"
    PKG_MANAGER=$(get_package_manager)
    
    if [ "$PKG_MANAGER" = "apt" ]; then
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
    elif [ "$PKG_MANAGER" = "dnf" ]; then
        sudo dnf install -y python3.10
    else
        echo -e "${RED}Unsupported package manager. Please install Python 3.10 manually.${NC}"
        exit 1
    fi
fi

# Ensure pip is available
if ! command_exists pip3; then
    echo -e "${YELLOW}Installing pip...${NC}"
    PKG_MANAGER=$(get_package_manager)
    
    if [ "$PKG_MANAGER" = "apt" ]; then
        sudo apt-get install -y python3-pip
    elif [ "$PKG_MANAGER" = "dnf" ]; then
        sudo dnf install -y python3-pip
    fi
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python3.10 -m pip install --upgrade pip

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Installing packages from requirements.txt...${NC}"
    if python3.10 -m pip install -r requirements.txt; then
        echo -e "${GREEN}Successfully installed all packages from requirements.txt${NC}"
    else
        echo -e "${RED}Error installing packages${NC}"
        read -n 1 -s -r -p "Press any key to exit..."
        exit 1
    fi
else
    echo -e "${RED}requirements.txt not found in the current directory!${NC}"
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

echo -e "${GREEN}Installation complete!${NC}"

read -n 1 -s -r -p "Press any key to exit..."
