import streamlit as st
import torch
import torchaudio
import gdown
import os
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample, MFCC
import soundfile as sf
import io
import torch.serialization

# Configuration
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400
HOP_LENGTH = 160
MODEL_URL = "https://drive.google.com/uc?id=1FdVrAZqoQ2Xz0GBEzDWTnexqWoX-oh6j"
MODEL_PATH = "best_model_safestudent.pth"
SUPPORTED_FORMATS = ['wav', 'mp3', 'flac', 'ogg', 'aac', 'm4a']

# Add safe globals for numpy scalar types
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

class AudioCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout(0.3)
        
        # Fixed dimensions based on your error message
        self.fc1 = torch.nn.Linear(7680, 512)  # Changed from 128*5*5 to 7680
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        
        # Flatten with correct dimensions
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model from Google Drive...'):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    
    # Load checkpoint to inspect number of classes
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    num_classes = checkpoint['teacher_state_dict']['fc2.weight'].shape[0]
    
    # Initialize model with correct dimensions
    model = AudioCNN(num_classes=num_classes)
    
    # Load state dict
    model.load_state_dict(checkpoint['teacher_state_dict'])
    model.eval()
    return model

@st.cache_data
def get_labels():
    # These should match exactly what was used during training
    return sorted([
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
        'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
        'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ])

def load_audio_file(uploaded_file):
    try:
        audio_bytes = uploaded_file.read()
        with io.BytesIO(audio_bytes) as f:
            data, sample_rate = sf.read(f)
        
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        return torch.from_numpy(data).float().unsqueeze(0), sample_rate
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        st.error("Supported formats: WAV, MP3, FLAC, OGG, AAC, M4A")
        return None, None

def preprocess_audio(waveform, sample_rate):
    if sample_rate != SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    if waveform.shape[1] < SAMPLE_RATE:
        waveform = torch.nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
    elif waveform.shape[1] > SAMPLE_RATE:
        waveform = waveform[:, :SAMPLE_RATE]
    
    mfcc_transform = MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'n_mels': 80,
            'center': False
        }
    )
    mfcc = mfcc_transform(waveform)
    
    if torch.isnan(mfcc).any() or torch.isinf(mfcc).any():
        mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return mfcc

def plot_waveform(waveform, sample_rate):
    plt.figure(figsize=(10, 3))
    plt.plot(waveform.numpy().T)
    plt.title("Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

def plot_top_predictions(probs, labels):
    top5_probs, top5_idxs = torch.topk(probs, 5)
    top5_labels = [labels[i] for i in top5_idxs[0].tolist()]
    top5_confidences = top5_probs[0].tolist()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(top5_labels[::-1], top5_confidences[::-1])
    ax.set_xlabel('Confidence')
    ax.set_title('Top 5 Predictions')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return fig

# Streamlit UI
st.title("🎤 Speech Command Recognition with SafeStudent")
st.write("Upload an audio file (MP3, WAV, FLAC, etc.) to classify the speech command")

uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=SUPPORTED_FORMATS,
    accept_multiple_files=False
)

if uploaded_file:
    try:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        waveform, sample_rate = load_audio_file(uploaded_file)
        if waveform is None:
            st.stop()
            
        duration = waveform.shape[1] / sample_rate
        if not (0.8 <= duration <= 1.5):
            st.warning(f"For best results, use 1-second audio. Current: {duration:.2f}s")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Duration: {duration:.2f} seconds")
        with col2:
            st.write(f"Sample rate: {sample_rate} Hz")
        
        plot_waveform(waveform, sample_rate)
        
        with st.spinner('Processing audio...'):
            features = preprocess_audio(waveform, sample_rate)
            
            model = load_model()
            labels = get_labels()
            
            with torch.no_grad():
                logits = model(features.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                top_prob, top_idx = torch.max(probs, dim=1)
        
        st.success(f"Predicted command: **{labels[top_idx]}** (confidence: {top_prob.item()*100:.1f}%)")
        
        fig = plot_top_predictions(probs, labels)
        st.pyplot(fig)
        
        with st.expander("Show all predictions"):
            for i, (label, prob) in enumerate(zip(labels, probs[0].tolist())):
                st.write(f"{label}: {prob*100:.2f}%")
                
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.error("Please ensure you've uploaded a valid audio file")
