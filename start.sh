#!/bin/bash

# Activate the Conda environment
if [ -x "$(command -v conda)" ]; then
    echo "Activating Conda environment 'huggingface'..."
    conda activate huggingface
else
    echo "Conda is not installed. Please install Conda to use this script."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the FastAPI server
echo "Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
