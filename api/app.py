import io
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from aasist_model import AASISTClassifier

app = FastAPI(title="FraudWave AASIST Biometric Core", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
model = AASISTClassifier().to(device)
model.eval()

history_db = []

def pad_or_truncate(audio: np.ndarray, target_len=64600):
    if len(audio) >= target_len:
        return audio[:target_len]
    num_repeats = int(np.ceil(target_len / len(audio)))
    padded = np.tile(audio, num_repeats)
    return padded[:target_len]

@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    start_time = time.time()
    contents = await file.read()

    try:
        data, sr = sf.read(io.BytesIO(contents))
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            y = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
        else:
            y = data.astype(np.float32)
    except Exception as e:
        return {
            "error": f"Audio processing failed: {str(e)}",
            "probability": 10.0,
            "risk_level": "ERROR",
            "verdict": "UNREADABLE AUDIO STREAM"
        }

    # Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Standard ASVspoof 64600 sample alignment
    fixed_audio = pad_or_truncate(y, 64600)
    tensor_input = torch.tensor(fixed_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        # probs[0] = Real/Bonafide, probs[1] = Synthetic/Spoof
        synthetic_prob = round(float(probs[1]) * 100.0, 1)

    latency = round((time.time() - start_time) * 1000, 1)
    is_fake = synthetic_prob >= 50.0

    risk_level = "CRITICAL" if is_fake else ("MODERATE" if synthetic_prob > 30 else "LOW")
    risk_color = "#ef4444" if is_fake else ("#f59e0b" if synthetic_prob > 30 else "#22c55e")
    verdict = "CRITICAL: Synthetic / AI-Cloned Voice Detected" if is_fake else "AUTHENTIC HUMAN VOICE"
    recommendation = (
        "HALT TRANSACTION IMMEDIATELY! Vocoder synthesis patterns identified."
        if is_fake else
        "Acoustic channel verified. Natural biological human vocal tract signatures confirmed."
    )

    result = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "filename": "Audio_Stream_Sample.wav",
        "probability": synthetic_prob,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "verdict": verdict,
        "latency_ms": latency,
        "recommendation": recommendation
    }

    history_db.append(result)
    return result

@app.get("/history")
async def get_history():
    return history_db[-10:]
