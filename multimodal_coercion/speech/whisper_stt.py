from typing import Dict, Any, Optional
from pathlib import Path
import tempfile


def convert_audio_to_wav(audio_path: str) -> str:
    """
    Convert any audio format to WAV format using librosa and soundfile.
    Returns path to converted WAV file.
    """
    import librosa
    import soundfile as sf

    p = Path(audio_path)
    ext = p.suffix.lower()

    # If already WAV, return as is
    if ext == ".wav":
        return audio_path

    try:
        # Load audio with librosa (supports mp3, m4a, flac, ogg, etc.)
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(temp_wav, y, sr)
        return temp_wav
    except Exception as e:
        raise RuntimeError(f"Failed to convert audio from {ext} to WAV: {e}")


from functools import lru_cache
import time

def _log_stage(name: str, start: float):
    now = time.time()
    print(f"[whisper] {name} took {now - start:.2f}s")
    return now


@lru_cache(maxsize=4)
def _load_whisper(model_name: str, device: str):
    import whisper
    print(f"loading whisper model {model_name} on {device}")
    return whisper.load_model(model_name, device=device)


def transcribe_tamil(
    audio_path: str, model_name: str = "base", device: str = "cpu"
) -> Dict[str, Any]:
    """Transcribe the given audio using Whisper.

    The model is cached in ``_load_whisper`` so repeated calls do not
    reload the weights.  We also log timing information so the caller can
    see which stages are slow.
    """
    t0 = time.time()
    # Convert audio to WAV if needed
    wav_path = convert_audio_to_wav(audio_path)
    t0 = _log_stage("audio conversion", t0)

    model = _load_whisper(model_name, device)
    t0 = _log_stage("model load", t0)

    # whisper.transcribe already chunks the audio; use small model set in
    # configuration for faster performance ("tiny" or "base" generally)
    fp16 = device.startswith("cuda")
    result = model.transcribe(
        wav_path,
        language="ta",
        task="transcribe",
        fp16=fp16,
        verbose=False,
        word_timestamps=False,
    )
    t0 = _log_stage("transcription", t0)

    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    # Clean up temp file if we created one
    if wav_path != audio_path:
        try:
            Path(wav_path).unlink()
        except Exception:
            pass
    return {"text": text, "segments": segments}
