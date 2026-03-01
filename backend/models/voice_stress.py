from typing import Optional
import numpy as np
import librosa


def voice_stress_score(wav_path: str, target_sr: int = 16000) -> float:
    """
    Compute a simple voice stress score in [0,1] using spectral and energy features.
    Higher value => more stressed.
    """
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    if y.size == 0:
        return 0.5
    # Energy and zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)[0]
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    # Pitch via autocorrelation proxy (f0 estimation is heavy; keep simple)
    # Normalize features
    zcr_norm = float(np.clip((np.mean(zcr) - 0.02) / 0.1, 0.0, 1.0))
    rms_var = float(np.var(rms))
    rms_norm = float(np.clip(rms_var / (np.mean(rms) ** 2 + 1e-6), 0.0, 1.0))
    # Combine (weights chosen conservatively)
    stress = 0.6 * zcr_norm + 0.4 * rms_norm
    return float(np.clip(stress, 0.0, 1.0))

