#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Create a virtual environment (optional but recommended)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt


# Run your Python app
python app.py
