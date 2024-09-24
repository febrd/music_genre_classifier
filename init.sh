#!/bin/bash

set -e
echo "Creating virtual environment..."
python3 -m venv music_genre_classifier_env
echo "Activating virtual environment..."
source music_genre_classifier_env/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip
echo "Installing dependencies..."
pip install -r requirements.txt
sudo apt-get install ffmpeg libsndfile1
echo "Starting Streamlit app..."
streamlit run app.py
