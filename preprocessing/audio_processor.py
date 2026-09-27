import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# Path to your audio file
audio_path = "sample_converted.wav"

# Load the audio
audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio length:", len(audio) / sample_rate, "seconds")

# Create Mel-Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate,
    n_mels=128
)

# Convert to decibels
mel_spectrogram_db = librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)

# Display Mel-Spectrogram
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_spectrogram_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("FraudWave - Mel-Spectrogram")
plt.tight_layout()

# Save image file to disk to verify it visually
plt.savefig("mel_spectrogram.png")  
print("Saved plot to mel_spectrogram.png")

# Display the window and block until closed
plt.show(block=True)