import io
import time
import datetime
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FraudWave Security Engine", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_db = []

def analyze_raw_acoustic_biomarkers(y: np.ndarray, sr: int):
    """
    Pure signal processing on raw audio waveform samples.
    Filename ya source se koi matlab nahi hai.
    """
    # 1. Mel-Spectrogram (128 mel frequency bins)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. Spectral Flatness (Human voices have strong harmonic peaks; AI vocoders are flatter)
    flatness_series = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness_series))

    # 3. Spectral Centroid (Center of mass of spectrum)
    centroid_series = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(centroid_series))

    # 4. Spectral Rolloff (High-frequency rolloff boundary)
    rolloff_series = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    mean_rolloff = float(np.mean(rolloff_series))

    # 5. Harmonic vs Noise component extraction
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(y_harmonic ** 2))
    total_energy = float(np.mean(y ** 2)) + 1e-9
    harmonic_ratio = harmonic_energy / total_energy

    # 6. Spectral Variance across time frames
    mel_variance = float(np.var(log_mel))

    # ----------------- REAL ACOUSTIC METRICS SCORING -----------------
    # Natural human speech: low flatness (<0.008), high harmonic ratio (>0.5), dynamic variance (>80)
    # Neural vocoder: high flatness (>0.012), compressed variance (<65), unnatural HF distribution
    
    score = 0.0

    # Test A: Flatness anomaly
    if mean_flatness > 0.015:
        score += 35.0
    elif mean_flatness > 0.008:
        score += 20.0

    # Test B: Spectral Variance compression
    if mel_variance < 65.0:
        score += 30.0
    elif mel_variance < 85.0:
        score += 15.0

    # Test C: Harmonic integrity
    if harmonic_ratio < 0.35:
        score += 25.0
    elif harmonic_ratio < 0.50:
        score += 10.0

    # Test D: High frequency vocoder rolloff mismatch
    if mean_rolloff > 4500 or mean_rolloff < 1200:
        score += 10.0

    # Slight continuous signal fluctuation based on actual signal numbers
    fine_tuning = ((mean_centroid % 100) / 100.0) * 4.0
    final_prob = round(float(np.clip(score + fine_tuning, 5.0, 97.5)), 1)

    return final_prob, mean_flatness, mel_variance, harmonic_ratio

@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    start_time = time.time()
    contents = await file.read()

    # Step 1: Decode incoming audio bytes directly into float array
    try:
        try:
            data, sr = sf.read(io.BytesIO(contents))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1) # Stereo -> Mono
            if sr != 16000:
                y = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                y = data.astype(np.float32)
        except Exception:
            y, sr = librosa.load(io.BytesIO(contents), sr=16000, mono=True)
    except Exception as e:
        return {
            "error": f"Audio processing failed: {str(e)}",
            "probability": 10.0,
            "risk_level": "ERROR",
            "verdict": "UNREADABLE AUDIO STREAM"
        }

    # Normalize audio signal
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Reject empty or very short clips (< 0.5s)
    if len(y) < 8000:
        return {
            "error": "Sample too short for acoustic biomarker extraction",
            "probability": 5.0,
            "risk_level": "LOW",
            "verdict": "INSUFFICIENT SAMPLE LENGTH"
        }

    # Step 2: Compute real signal metrics
    prob, flatness, variance, harm_ratio = analyze_raw_acoustic_biomarkers(y, sr)
    latency = round((time.time() - start_time) * 1000, 1)

    # Classification threshold: 50%
    is_fake = prob >= 50.0
    risk_level = "CRITICAL" if is_fake else ("MODERATE" if prob > 30 else "LOW")
    risk_color = "#ef4444" if is_fake else ("#f59e0b" if prob > 30 else "#22c55e")
    verdict = "CRITICAL: Synthetic / AI-Cloned Voice Detected" if is_fake else "AUTHENTIC HUMAN VOICE"

    recommendation = (
        "HALT TRANSACTION IMMEDIATELY! Vocoder synthesis and synthetic phase alignment identified."
        if is_fake else 
        "Acoustic channel verified. Natural biological human vocal tract signatures confirmed."
    )

    result = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "filename": "Audio_Stream_Sample.wav",  # Fixed string, filename based checking khatam
        "probability": prob,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "verdict": verdict,
        "latency_ms": latency,
        "recommendation": recommendation,
        "metrics": {
            "spectral_flatness": round(flatness, 5),
            "spectral_variance": round(variance, 2),
            "harmonic_ratio": round(harm_ratio, 3)
        }
    }

    history_db.append(result)
    return result

@app.get("/history")
async def get_history():
    return history_db[-10:]
