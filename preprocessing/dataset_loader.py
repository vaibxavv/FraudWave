import os
import numpy as np
import librosa

# Configuration
DATASET_PATH = "dataset"
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
N_MELS = 128

def process_file(file_path):
    """Loads an audio file, pads/trims to fixed length, and extracts Mel-spectrogram."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # Pad if shorter than target length, trim if longer
    if len(audio) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")
    else:
        audio = audio[:TARGET_LENGTH]
        
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize between 0 and 1
    mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_norm

def load_dataset():
    """Iterates through real and fake directories and returns arrays and labels."""
    X = []
    y = []
    
    classes = {"real": 0, "fake": 1}
    
    for label_name, label_idx in classes.items():
        folder = os.path.join(DATASET_PATH, label_name)
        if not os.path.exists(folder):
            continue
            
        for file in os.listdir(folder):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(folder, file)
                try:
                    features = process_file(file_path)
                    X.append(features)
                    y.append(label_idx)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
                    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset loaded: {len(X)} samples total.")
    if len(X) > 0:
        print(f"Feature shape per sample: {X[0].shape}")
        print(f"Real samples (0): {np.sum(y == 0)}")
        print(f"Fake samples (1): {np.sum(y == 1)}")
        
    return X, y

if __name__ == "__main__":
    X, y = load_dataset()