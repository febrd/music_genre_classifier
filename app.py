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


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sr' not in st.session_state:
    st.session_state.sr = None
if 'genre' not in st.session_state:
    st.session_state.genre = None
if 'prediction_mode' not in st.session_state:
    st.session_state.prediction_mode = None

def ai_predict_genre(audio_data):
    audio_input = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(audio_input.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)  
    transcription = transcription[0]  

    genre_mapping = {
        "pop": "Pop",
        "rock": "Rock",
        "electronic": "Electronic",
        "classical": "Classical",
        "jazz": "Jazz",
        "metal": "Metal",
        "reggae": "Reggae",
        "blues": "Blues",
        "folk": "Folk",
        "hip-hop": "Hip-Hop",
        "ambient": "Ambient"
    }
    return genre_mapping.get(transcription.lower(), "Other")

def extract_audio_features(y):
    if len(y) < 512: 
        st.warning("Audio signal is too short for feature extraction.")
        return None
    
    features = {}
    features['tempo'], _ = librosa.beat.beat_track(y=y, sr=16000)
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=16000, n_fft=512))
    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512))
    features['rmse'] = np.mean(librosa.feature.rms(y=y, frame_length=2048, hop_length=512))
    features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=y, sr=16000))
    features['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=16000, n_fft=512))
    features['mfccs'] = np.mean(librosa.feature.mfcc(y=y, n_mfcc=13, sr=16000), axis=0)
    
    return features



def predict_genre(features, mode='simple'):
    if mode == 'simple':
        conditions = [
            (features['tempo'] < 100 and features['spectral_centroid'] < 2000 and features['zero_crossing_rate'] < 0.1, "Pop"),
            (100 <= features['tempo'] < 130 and features['rmse'] > 0.05, "Rock"),
            (features['tempo'] >= 130 and features['tonnetz'] < -0.1, "Electronic"),
            (features['tempo'] < 80 and features['spectral_centroid'] > 3000, "Classical"),
            (90 <= features['tempo'] < 110 and features['chroma_stft'] > 0.5, "Reggae"),
            (70 <= features['tempo'] < 90 and np.mean(features['mfccs']) < -300, "Blues"),
            (features['tempo'] >= 110 and np.mean(features['mfccs']) > -200 and features['spectral_centroid'] < 2500, "Country"),
            (features['tempo'] >= 120 and features['tonnetz'] > 0.2, "Jazz"),
            (features['tempo'] >= 140 and features['rmse'] > 0.1, "Metal"),
            (60 <= features['tempo'] < 80 and features['spectral_centroid'] < 1500 and features['zero_crossing_rate'] > 0.1, "Folk"),
            (features['tempo'] >= 90 and features['spectral_centroid'] < 3000 and features['tonnetz'] < -0.2, "Hip-Hop"),
            (features['tempo'] < 60 and features['zero_crossing_rate'] < 0.05, "Ambient"),
        ]
        for condition, genre in conditions:
            if condition:
                return genre
        return "Other"
    elif mode == 'ai':
        return ai_predict_genre(y)

def change_audio_speed(audio_segment, speed=1.0):
    return audio_segment.speedup(playback_speed=speed)

def convert_audio_format(uploaded_file):
    try:
        audio_segment = AudioSegment.from_file(uploaded_file)
        wav_buffer = BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer
    except CouldntDecodeError as e:
        st.error("Could not decode the audio file. Please upload a valid MP3 or WAV file.")
        st.write(f"Error details: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred during conversion: {str(e)}")
        return None

def process_audio(file):  
    wav_file = convert_audio_format(file)  
    if wav_file is None:  
        return None, None  

    try:  
        y, sr = librosa.load(wav_file, sr=16000)  # Use a consistent sample rate

        num_samples = 2000  
        if len(y) > num_samples:
            y = y[::len(y) // num_samples]  

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.linspace(0, len(y)/sr, num=len(y)), y=y, mode='lines', name='Waveform'))
        fig.update_layout(title='Waveform of the Audio', xaxis_title='Time (seconds)', yaxis_title='Amplitude', height=400)
        st.plotly_chart(fig)

        features = extract_audio_features(y)
        if st.session_state.prediction_mode == 'AI Predicted':
            st.session_state.genre = ai_predict_genre(y)
        else:  
            st.session_state.genre = predict_genre(features, mode='simple')

        st.write(f"**Predicted Genre**: {st.session_state.genre}")
        st.write(f"**Predicted Mode**: {st.session_state.prediction_mode}")

        return y, sr  
    except Exception as e:  
        st.error(f"Error loading audio with librosa: {str(e)}")  
        return None, None  

def play_audio(audio_segment):
    audio_buffer = BytesIO()
    audio_segment.export(audio_buffer, format="wav")
    st.audio(audio_buffer)

st.set_page_config(page_title="Music Genre Classifier", layout="wide")
st.markdown("<h1 style='text-align: center;'>Music Classifier - <span style='color: blue;'>AI Powered</span></h1>", unsafe_allow_html=True)
st.session_state.prediction_mode = st.selectbox("Choose a Prediction Model", ["Select Prediction Model", "Simple", "AI Predicted"])

if st.session_state.prediction_mode and st.session_state.prediction_mode != "Select Prediction Model":
    SAMPLE_DIR = 'SAMPLE'
    audio_files = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.endswith(('mp3', 'wav', 'ogg'))]

    selected_file = st.selectbox("Select a music file", ["Choose audio first"] + audio_files)

    if selected_file != "Choose audio first":
        st.session_state.audio_data = None
        st.session_state.sr = None
        st.session_state.genre = None

        try:
            audio_data = AudioSegment.from_file(selected_file)
            st.session_state.audio_data = audio_data  # Store audio data in session state
            st.write("Original Audio:")
            play_audio(audio_data)

            speed = st.slider("Select Playback Speed", 0.5, 4.0, 1.0, step=0.1)
            modified_audio = change_audio_speed(audio_data, speed=speed)
            st.write(f"Audio at {speed}x speed:")
            play_audio(modified_audio)
            y, sr = process_audio(selected_file)
            st.session_state.y = y  
            st.session_state.sr = sr  

        except CouldntDecodeError:
            st.error("Could not decode the audio file. Please upload a valid MP3, WAV, or OGG file.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

st.markdown("""
    <style>
        footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #333; /* Dark background */
            color: white;           /* White text */
            text-align: center;
            padding: 20px 0;       /* Increased padding */
            font-size: 14px;        /* Font size */
            box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.3); /* Subtle shadow */
        }
        footer p {
            margin: 0; /* Remove default margin */
        }
    </style>
    <footer>
        <p>© 2024 - Febriansah Dirgantara. All rights reserved.</p>
    </footer>
""", unsafe_allow_html=True)