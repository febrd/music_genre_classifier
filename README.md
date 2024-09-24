# Music Genre Classifier

This project implements a music genre classification application using Streamlit, leveraging audio processing and machine learning techniques. The application allows users to upload audio files and predicts their genres using a combination of traditional audio feature extraction and an AI model based on Wav2Vec2.

## Table of Contents
- [Overview](#overview)
- [Libraries Used](#libraries-used)
- [How to Run the Application](#how-to-run-the-application)
- [Application Structure](#application-structure)
- [Code Explanation](#code-explanation)

## Overview

The Music Genre Classifier is a Streamlit application that allows users to:
1. Upload audio files in various formats (MP3, WAV, OGG).
2. Select a prediction mode (simple rules-based or AI-based).
3. Visualize the audio waveform.
4. Modify playback speed and listen to the audio.
5. Predict the genre of the audio based on its features.

## Libraries Used

The following libraries are used in this project:

- **Streamlit**: A framework for building interactive web applications in Python. It is used to create the user interface.
- **Librosa**: A Python library for music and audio analysis. It provides functionalities for feature extraction and audio manipulation.
- **NumPy**: A library for numerical computations in Python. It is used for handling arrays and performing mathematical operations.
- **Plotly**: A library for creating interactive visualizations. It is used to plot the audio waveform.
- **Pydub**: A library for audio manipulation. It is used to handle different audio formats and playback speed modifications.
- **Torch**: A deep learning library used for building and training neural networks. It is used here for loading and using the Wav2Vec2 model.
- **Transformers**: A library by Hugging Face that provides pre-trained models for natural language processing tasks. It is used to load the Wav2Vec2 model for audio classification.

## How to Run the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-genre-classifier.git
   cd music-genre-classifier
   ```

2. Run & Install the required libraries:
   ```bash
   chmod +x init.sh
   ./init.sh
   ```

4. Open the application in your web browser at `http://localhost:8501`.

## Application Structure

The main application code is located in `app.py`. It includes:
- Audio processing functions.
- Genre prediction logic.
- Streamlit UI components.

## Code Explanation

### Importing Libraries

```python
import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from io import BytesIO
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
```

- **streamlit**: Used to create the web application.
- **librosa**: For audio processing and feature extraction.
- **numpy**: For numerical operations on audio data.
- **plotly**: For plotting the audio waveform.
- **pydub**: For handling audio files and playback speed modification.
- **torch**: For using the Wav2Vec2 model.
- **transformers**: To load pre-trained models from Hugging Face.

### Load Model and Processor

```python
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
```

- Loads a pre-trained Wav2Vec2 model and processor, which are used for audio transcription and genre prediction.

### Initialize Session State

```python
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sr' not in st.session_state:
    st.session_state.sr = None
if 'genre' not in st.session_state:
    st.session_state.genre = None
if 'prediction_mode' not in st.session_state:
    st.session_state.prediction_mode = None
```

- Initializes session state variables to store audio data, sample rate, predicted genre, and prediction mode.

### Audio Prediction Functions

#### AI Genre Prediction

```python
def ai_predict_genre(audio_data):
    audio_input = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(audio_input.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    genre_mapping = { ... }
    return genre_mapping.get(transcription.lower(), "Other")
```

- Processes the audio data, feeds it to the Wav2Vec2 model, and decodes the predicted transcription to determine the genre.

#### Feature Extraction

```python
def extract_audio_features(y):
    features = {}
    features['tempo'], _ = librosa.beat.beat_track(y=y)
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y))
    ...
    return features
```

- Extracts various audio features such as tempo, spectral centroid, zero crossing rate, and MFCCs from the audio signal.

#### Genre Prediction Logic

```python
def predict_genre(features, mode='simple'):
    if mode == 'simple':
        conditions = [ ... ]
        for condition, genre in conditions:
            if condition:
                return genre
        return "Other"
    elif mode == 'ai':
        return ai_predict_genre(y)
```

- Determines the genre based on extracted features using either a simple rules-based approach or AI-based prediction.

### Audio Processing Functions

#### Convert Audio Format

```python
def convert_audio_format(uploaded_file):
    ...
    return wav_buffer
```

- Converts uploaded audio files to WAV format for consistent processing.

#### Process Audio

```python
def process_audio(file):  
    wav_file = convert_audio_format(file)  
    ...
    return y, sr  
```

- Loads and processes the audio file, extracting features and visualizing the waveform.

### Streamlit User Interface

```python
st.set_page_config(page_title="Music Genre Classifier", layout="wide")
st.title("Advanced Music Genre Classifier with Speed Control")
```

- Sets up the Streamlit application layout and title.

### Running the Application

The main part of the application is structured with conditional logic to handle user inputs, playback, and genre prediction. The application listens for audio uploads, allows users to select playback speed, and displays the predicted genre based on the selected prediction mode.

## Conclusion

This Music Genre Classifier application is a powerful tool for analyzing and categorizing audio files. It showcases the integration of audio processing techniques and machine learning models to deliver insightful genre predictions in a user-friendly interface.
```
