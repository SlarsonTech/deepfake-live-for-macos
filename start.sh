#!/bin/bash

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Running setup first..."
    ./setup.sh
    source venv/bin/activate
fi

# Run the application with CoreML
echo "Starting Deepfake Live Camera..."
python3.10 run.py --execution-provider coreml 