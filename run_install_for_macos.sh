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

# Install Homebrew if not installed
if ! command_exists brew; then
    echo -e "${YELLOW}Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install Python 3.10 if not installed
if ! command_exists python3; then
    echo -e "${YELLOW}Installing Python 3.10...${NC}"
    brew install python@3.10
    brew link python@3.10
fi

# Ensure pip is available
if ! command_exists pip3; then
    echo -e "${YELLOW}Installing pip...${NC}"
    python3 -m ensurepip --default-pip
fi

# Install pip requirements
echo -e "${YELLOW}Installing pip requirements...${NC}"
pip3 install --upgrade pip

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Installing packages from requirements.txt...${NC}"
    if pip3 install -r requirements.txt; then
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
