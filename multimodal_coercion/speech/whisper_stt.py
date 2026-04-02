"""
Whisper Speech-to-Text with transcription confidence estimation.

Extracts confidence scores from Whisper's segment-level log probabilities
to assess transcription reliability.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import numpy as np


@dataclass
class TranscriptionResult:
    """
    Result of speech-to-text transcription with confidence estimates.
    
    Attributes:
        text: Transcribed text
        confidence: Average confidence score (0-1)
        is_reliable: Whether transcription passes confidence threshold
        segments: Raw Whisper segments with timing information
        segment_confidences: Confidence score for each segment
    """
    text: str
    confidence: float
    is_reliable: bool
    segments: List[Dict[str, Any]] = None
    segment_confidences: List[float] = None

    def __post_init__(self):
        """Ensure confidence is in valid range."""
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        if self.segments is None:
            self.segments = []
        if self.segment_confidences is None:
            self.segment_confidences = []


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


def extract_segment_confidence(segment: Dict[str, Any]) -> float:
    """
    Extract average confidence from a Whisper segment.
    
    Whisper provides per-token log probabilities in segment['tokens'].
    We convert these to probabilities and average them.
    
    Args:
        segment: Whisper segment dict with optional 'tokens' field
        
    Returns:
        Confidence score (0-1). Default 0.5 if no token data available.
    """
    # Try to extract probabilities from tokens
    tokens = segment.get("tokens", [])
    
    if not tokens:
        # No token data, use default confidence
        return 0.5
    
    try:
        # Extract logprobs from tokens
        logprobs = []
        for token in tokens:
            if isinstance(token, dict) and "logprob" in token:
                logprobs.append(token["logprob"])
        
        if not logprobs:
            return 0.5
        
        # Convert log probabilities to probabilities
        # logprob is typically negative, so take exp(logprob)
        logprobs = np.array(logprobs)
        
        # Clip to reasonable range to avoid numerical issues
        logprobs = np.clip(logprobs, -20, 0)
        
        # Convert to probabilities
        probs = np.exp(logprobs)
        
        # Average probability
        confidence = float(np.mean(probs))
        return np.clip(confidence, 0.0, 1.0)
    
    except Exception as e:
        # If extraction fails, use default
        print(f"[whisper] Warning: failed to extract confidence: {e}")
        return 0.5


def transcribe_tamil(
    audio_path: str, model_name: str = "base", device: str = "cpu",
    confidence_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Transcribe Tamil audio using Whisper with confidence estimation.
    
    The model is cached so repeated calls do not reload the weights.
    Extracts confidence from segment-level log probabilities.
    
    Args:
        audio_path: Path to audio file (any format)
        model_name: Whisper model size ("tiny", "base", "small", etc.)
        device: Device to use ("cpu", "cuda", etc.)
        confidence_threshold: Confidence threshold for is_reliable flag (default 0.6)
        
    Returns:
        Dict with keys:
        - text: Transcribed text
        - nlp_prob: NLP coercion probability (NLP classifier score)
        - confidence: Average transcription confidence
        - is_reliable: Whether transcription passes confidence threshold
        - segments: Whisper segment data
        - segment_confidences: Per-segment confidence scores
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

    # Extract confidence from each segment
    segment_confidences = [extract_segment_confidence(seg) for seg in segments]
    
    # Compute average confidence
    if segment_confidences:
        avg_confidence = float(np.mean(segment_confidences))
    else:
        avg_confidence = 0.5

    # Determine if reliable (passes threshold)
    is_reliable = avg_confidence >= confidence_threshold

    # Clean up temp file if we created one
    if wav_path != audio_path:
        try:
            Path(wav_path).unlink()
        except Exception:
            pass

    # Return both old format (for backward compatibility) and new format
    return {
        "text": text,
        "nlp_prob": None,  # Will be filled by NLP pipeline
        "confidence": avg_confidence,
        "is_reliable": is_reliable,
        "segments": segments,
        "segment_confidences": segment_confidences,
    }


def transcribe_tamil_with_result(
    audio_path: str, model_name: str = "base", device: str = "cpu",
    confidence_threshold: float = 0.6
) -> TranscriptionResult:
    """
    Transcribe Tamil audio and return TranscriptionResult object.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size
        device: Device to use
        confidence_threshold: Confidence threshold for is_reliable
        
    Returns:
        TranscriptionResult with transcription and confidence data
    """
    result_dict = transcribe_tamil(audio_path, model_name, device, confidence_threshold)
    
    return TranscriptionResult(
        text=result_dict["text"],
        confidence=result_dict["confidence"],
        is_reliable=result_dict["is_reliable"],
        segments=result_dict.get("segments", []),
        segment_confidences=result_dict.get("segment_confidences", []),
    )

