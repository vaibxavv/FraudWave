import io
import time
import datetime
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FraudWave Biometric Core", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_db = []

def extract_biometric_features(y: np.ndarray, sr: int):
    # 1. Noise gate & Trim
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trim) > sr * 0.5:
        y = y_trim

    # 2. MFCC & Delta analysis (Biological vocal tract dynamics)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    delta_variance = float(np.mean(np.var(mfcc_delta, axis=1)))

    # 3. Spectral Flatness
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # 4. Zero Crossing Rate (ZCR) Stability
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = float(np.mean(zcr))

    # 5. Harmonic to Percussive Ratio
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_power = float(np.mean(y_harm ** 2))
    total_power = float(np.mean(y ** 2)) + 1e-9
    hnr = harm_power / total_power

    # ------------------- ROBUST DISCRIMINATOR -------------------
    # Real speech has high delta dynamics (natural speech transitions) and low spectral flatness
    # Cloned/TTS speech has compressed delta dynamics or unnatural vocoder buzz
    
    score = 18.0  # Base natural voice anchor

    # Delta Variance check (Humans > 1.8, Vocoders often < 1.2 or > 3.5 unnatural)
    if delta_variance < 1.3:
        score += 35.0  # Robotic static transitions
    elif delta_variance > 3.8:
        score += 30.0  # Glitchy phase vocoder artifacts

    # Flatness check (Natural vocal cords have distinct harmonic peaks)
    if flatness > 0.02:
        score += 25.0
    elif flatness < 0.005:
        score -= 6.0   # Clear biological resonance

    # Harmonic richness check
    if hnr < 0.30:
        score += 20.0
    elif hnr > 0.55:
        score -= 8.0   # Strong vocal resonance reward

    # ZCR check
    if zcr_mean > 0.12:
        score += 15.0

    final_prob = round(float(np.clip(score, 14.2, 94.6)), 1)
    return final_prob

@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    start_time = time.time()
    contents = await file.read()

    try:
        try:
            data, sr = sf.read(io.BytesIO(contents))
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if sr != 16000:
                y = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                y = data.astype(np.float32)
        except Exception:
            y, sr = librosa.load(io.BytesIO(contents), sr=16000, mono=True)
    except Exception as e:
        return {"error": f"Audio processing failed: {str(e)}", "probability": 15.0, "risk_level": "ERROR"}

    if len(y) < 8000:
        return {"error": "Audio too short", "probability": 15.0, "risk_level": "LOW", "verdict": "INSUFFICIENT LENGTH"}

    prob = extract_biometric_features(y, sr)
    latency = round((time.time() - start_time) * 1000, 1)

    is_fake = prob >= 50.0
    risk_level = "CRITICAL" if is_fake else ("MODERATE" if prob > 30 else "LOW")
    risk_color = "#ef4444" if is_fake else ("#f59e0b" if prob > 30 else "#22c55e")
    verdict = "CRITICAL: Synthetic / AI-Cloned Voice Detected" if is_fake else "AUTHENTIC HUMAN VOICE"

    result = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "filename": file.filename or "audio_sample.wav",
        "probability": prob,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "verdict": verdict,
        "latency_ms": latency
    }

    history_db.append(result)
    return result

@app.get("/history")
async def get_history():
    return history_db[-10:]
