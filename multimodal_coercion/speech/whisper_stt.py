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


def transcribe_tamil(
    audio_path: str, model_name: str = "base", device: str = "cpu"
) -> Dict[str, Any]:
    import whisper

    # Convert audio to WAV if needed
    wav_path = convert_audio_to_wav(audio_path)

    model = whisper.load_model(model_name, device=device)
    fp16 = device.startswith("cuda")
    result = model.transcribe(wav_path, language="ta", task="transcribe", fp16=fp16)
    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    # Clean up temp file if we created one
    if wav_path != audio_path:
        try:
            Path(wav_path).unlink()
        except Exception:
            pass

    return {"text": text, "segments": segments}
