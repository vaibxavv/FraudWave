import io
import time
import datetime
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FraudWave Core", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history_db = []

def analyze_audio_biometrics(y: np.ndarray, sr: int):
    # 1. Trim silence
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    if len(y_trim) > sr * 0.4:
        y = y_trim

    # 2. Extract Spectral Centroid & Flatness
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    # 3. Fundamental Frequency (F0) Tracking
    f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr)
    voiced = f0[~np.isnan(f0)]
    f0_std = float(np.std(voiced)) if len(voiced) > 8 else 0.0

    # 4. Harmonic vs Percussive ratio
    harm, _ = librosa.effects.hpss(y)
    harm_energy = float(np.mean(harm ** 2))
    total_energy = float(np.mean(y ** 2)) + 1e-9
    harm_ratio = harm_energy / total_energy

    # 5. High-Frequency Rolloff
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))

    # --- Calibrated Heuristic Weights ---
    score = 15.0

    # Human voice has rich pitch fluctuations (f0_std > 18 Hz)
    # AI vocoders are strictly pitch-smoothed or quantized (f0_std < 12 Hz)
    if len(voiced) > 8:
        if f0_std < 10.0:
            score += 35.0
        elif f0_std < 16.0:
            score += 18.0
        elif f0_std > 24.0:
            score -= 10.0

    # AI speech has higher spectral flatness across high frequencies
    if flatness > 0.018:
        score += 28.0
    elif flatness < 0.006:
        score -= 8.0

    # Vocoder phase bounds
    if 3400 < rolloff < 5200:
        score += 20.0

    if harm_ratio < 0.25:
        score += 15.0
    elif harm_ratio > 0.50:
        score -= 8.0

    final_prob = round(float(np.clip(score, 12.0, 94.0)), 1)
    return final_prob

@app.get("/")
def root():
    return {"service": "FraudWave", "status": "running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()

    try:
        try:
            data, sr = sf.read(io.BytesIO(raw))
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                y = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                y = data.astype(np.float32)
        except Exception:
            y, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
    except Exception as e:
        return {"error": f"Audio decode failed: {str(e)}", "probability": 15.0, "risk_level": "LOW"}

    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    if len(y) < 6000:
        return {"error": "Audio too short", "probability": 15.0, "risk_level": "LOW", "verdict": "TOO SHORT"}

    prob = analyze_audio_biometrics(y, sr)
    latency = round((time.time() - t0) * 1000, 1)

    is_fake = prob >= 50.0
    level = "CRITICAL" if is_fake else ("MODERATE" if prob > 30 else "LOW")
    color = "#ef4444" if is_fake else ("#f59e0b" if prob > 30 else "#22c55e")
    verdict = "CRITICAL: Synthetic / AI-Cloned Voice Detected" if is_fake else "AUTHENTIC HUMAN VOICE"

    rec = (
        "AI voice characteristics identified. Verify speaker identity."
        if is_fake else
        "Natural vocal tract patterns verified. Authentic speech."
    )

    res = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "filename": file.filename or "audio_sample.wav",
        "probability": prob,
        "risk_level": level,
        "risk_color": color,
        "verdict": verdict,
        "latency_ms": latency,
        "recommendation": rec
    }

    history_db.append(res)
    return res

@app.get("/history")
def history():
    return history_db[-10:]
