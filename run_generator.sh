#!/bin/bash

# Synthetic Data Generator Runner
# This script provides a simple way to run the Synthetic Data Generator

echo "======================================"
echo "Synthetic Data Generator Quick Starter"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "No virtual environment found. Creating one..."
    python3 -m venv .venv
    
    # Activate virtual environment
    source .venv/bin/activate
    
    echo "Installing required dependencies..."
    pip install -r requirements.txt
    
    echo "Environment setup complete!"
else
    # Activate virtual environment
    source .venv/bin/activate
    echo "Using existing virtual environment."
fi

echo ""
echo "Starting Synthetic Data Generator..."
echo ""

# Run the generator
python synthetic_data_generator.py

# Deactivate virtual environment
deactivate

echo ""
echo "Generator session completed. Exiting..."
