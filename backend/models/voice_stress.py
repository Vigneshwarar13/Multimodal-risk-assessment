from typing import Optional
import time
import numpy as np
import librosa


def _log_stage(name: str, start: float):
    now = time.time()
    print(f"[voice_stress] {name} took {now - start:.2f}s")
    return now


def voice_stress_score(wav_path: str, target_sr: int = 16000) -> float:
    """Compute voice stress score using spectral and energy features.

    This function uses zero-crossing rate, RMS energy variance, and other
    simple features to avoid expensive pitch estimation (Yad autocorrelation).
    The features are combined using fixed weights.
    """
    t0 = time.time()
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    t0 = _log_stage("audio load", t0)

    if y.size == 0:
        return 0.5
    # Energy and zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)[0]
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    t0 = _log_stage("feature extraction", t0)

    # Normalize features
    zcr_norm = float(np.clip((np.mean(zcr) - 0.02) / 0.1, 0.0, 1.0))
    rms_var = float(np.var(rms))
    rms_norm = float(np.clip(rms_var / (np.mean(rms) ** 2 + 1e-6), 0.0, 1.0))
    # Combine (weights chosen conservatively)
    stress = 0.6 * zcr_norm + 0.4 * rms_norm
    t0 = _log_stage("normalization", t0)
    return float(np.clip(stress, 0.0, 1.0))

