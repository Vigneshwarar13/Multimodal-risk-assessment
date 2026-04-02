from typing import Dict, Any
from pathlib import Path
from multimodal_coercion.speech.whisper_stt import transcribe_tamil
from multimodal_coercion.speech.text_preprocess import clean_tamil_text
from multimodal_coercion.speech.nlp_classifier import TamilCoercionClassifier
from multimodal_coercion.core.config import Config, project_root, get_config


from functools import lru_cache
import time


def _log_stage(name: str, start: float):
    now = time.time()
    print(f"[speech_pipeline] {name} took {now - start:.2f}s")
    return now


# cache classifiers keyed by model name/path
_classifier_cache = {}


def _get_classifier(name: str):
    if name not in _classifier_cache:
        clf = TamilCoercionClassifier(name)
        clf.load()
        _classifier_cache[name] = clf
    return _classifier_cache[name]


@lru_cache(maxsize=4)
def _whisper_params(model_name: str, device: str):
    return model_name, device


def run_speech_pipeline(audio_path: str) -> Dict[str, Any]:
    """
    Run speech processing pipeline: STT → text cleaning → NLP coercion classification.
    
    Returns:
        Dict with:
        - transcript: Cleaned transcribed text
        - nlp_prob: NLP coercion probability (0-1)
        - label: Coercion label (Genuine Consent / Neutral / Coercion)
        - transcription_confidence: Whisper confidence (0-1)
        - transcription_reliable: Whether confidence passes threshold
        - timestamps: Empty list (for future per-token timing)
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    t0 = time.time()
    base = project_root()
    cfg = get_config(base)

    asr_cfg = cfg.models.get("whisper", {})
    model_name = asr_cfg.get("model", "base")
    device = asr_cfg.get("device", "cpu")
    confidence_threshold = asr_cfg.get("confidence_threshold", 0.6)
    
    # Transcribe with confidence estimation
    stt = transcribe_tamil(
        str(p),
        model_name=model_name,
        device=device,
        confidence_threshold=confidence_threshold
    )
    t0 = _log_stage("asr", t0)

    transcript = clean_tamil_text(stt.get("text", ""))
    t0 = _log_stage("clean_text", t0)

    nlp_path = cfg.models.get("nlp_tamil", {}).get("path")
    nlp_model = cfg.models.get("nlp_tamil", {}).get("model")
    clf_name = nlp_path if nlp_path else (nlp_model or "ai4bharat/indic-bert")
    classifier = _get_classifier(clf_name)
    coercion_prob, label = classifier.predict(transcript or " ")
    t0 = _log_stage("nlp", t0)

    return {
        "transcript": transcript,
        "nlp_prob": float(coercion_prob),
        "label": label,
        "transcription_confidence": stt.get("confidence", 0.5),
        "transcription_reliable": stt.get("is_reliable", False),
        "timestamps": [],
    }
