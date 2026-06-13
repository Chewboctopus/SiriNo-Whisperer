#!/bin/bash
echo "🚀 Welcome to MacWhisperer Installer!"
echo "This script will automatically set up everything you need."
echo "---------------------------------------------------------"

# Get directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Homebrew not found. Installing Homebrew (this may ask for your Mac password)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add brew to path for Apple Silicon
    if [[ -d /opt/homebrew/bin ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "✅ Homebrew is already installed."
fi

echo "📦 Installing required system audio libraries (ffmpeg, portaudio)..."
brew install portaudio ffmpeg

echo "🐍 Setting up isolated Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "⬇️ Installing AI dependencies (this may take a minute)..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "---------------------------------------------------------"
echo "🎉 INSTALLATION COMPLETE! 🎉"
echo "You can safely close this terminal window."
echo ""
echo "👉 TO START: Double-click 'start_whisper.command' in Finder!"
