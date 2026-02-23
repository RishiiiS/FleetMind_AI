#!/bin/bash

set -e

echo "=========================================="
echo "    FleetMind AI - Setup & Execution      "
echo "=========================================="

echo -e "\n[1/4] Creating an isolated Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo -e "\n[2/4] Installing required Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "\n[3/4] Preparing AI Model..."
if [ ! -f "fleetmind_model.pkl" ]; then
    echo "Model file not found. Downloading dataset and training the model now..."
    python fleetmind_ai.py
else
    echo "fleetmind_model.pkl already exists. Skipping training."
fi

echo -e "\n[4/4] Starting the Streamlit interactive dashboard..."
streamlit run app.py