#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Starting MacWhisperer..."
echo "Note: If hotkeys do not work, please ensure Terminal/iTerm has 'Accessibility' permissions in System Settings -> Privacy & Security."

# Configuration
# Keys are now securely loaded from the .env file!

# Activate virtual environment
source venv/bin/activate

# Run the python script
python main.py
