def analyze_raw_acoustic_biomarkers(y: np.ndarray, sr: int):
    # 1. Background Noise / Silence Trimming (Crucial for laptop mics)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) > sr * 0.5:
        y = y_trimmed

    # 2. Mel-Spectrogram (128 bins)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Spectral Flatness (Human speech: very low, AI: elevated)
    flatness_series = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness_series))

    # 4. Spectral Variance (Dynamic range across mel bins)
    mel_variance = float(np.var(log_mel))

    # 5. Harmonic Energy vs Total Energy
    y_harmonic, _ = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(y_harmonic ** 2))
    total_energy = float(np.mean(y ** 2)) + 1e-9
    harmonic_ratio = harmonic_energy / total_energy

    # 6. Fundamental Frequency (F0 / Pitch Variation)
    # Natural human voice has rich pitch shifts; AI TTS often has flat/mechanical F0
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_voiced = f0[~np.isnan(f0)]
    f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 10 else 0.0

    # ------------------ CALIBRATED BIOMARKER SCORING ------------------
    base_score = 10.0

    # Rule 1: Spectral Flatness (Higher thresholds to avoid mic noise penalty)
    if mean_flatness > 0.025:
        base_score += 35.0
    elif mean_flatness > 0.015:
        base_score += 20.0

    # Rule 2: Pitch Dynamic Range (Std Dev of F0)
    # In humans, f0_std is usually > 18 Hz due to expressive inflection
    if f0_std < 12.0 and len(f0_voiced) > 10:
        base_score += 25.0
    elif f0_std > 22.0:
        base_score -= 10.0 # Reward natural human inflection

    # Rule 3: Harmonic Integrity
    if harmonic_ratio < 0.25:
        base_score += 20.0
    elif harmonic_ratio > 0.45:
        base_score -= 10.0 # Reward strong biological vocal tract resonance

    # Rule 4: Mel Variance Compression
    if mel_variance < 55.0:
        base_score += 20.0

    final_prob = round(float(np.clip(base_score, 8.5, 96.5)), 1)
    return final_prob, mean_flatness, mel_variance, harmonic_ratio
