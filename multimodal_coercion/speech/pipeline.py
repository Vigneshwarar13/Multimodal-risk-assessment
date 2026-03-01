from typing import Dict, Any
from pathlib import Path
from multimodal_coercion.speech.whisper_stt import transcribe_tamil
from multimodal_coercion.speech.text_preprocess import clean_tamil_text
from multimodal_coercion.speech.nlp_classifier import TamilCoercionClassifier
from multimodal_coercion.core.config import Config, project_root


def run_speech_pipeline(audio_path: str) -> Dict[str, Any]:
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    base = project_root()
    cfg = Config(base)
    asr_cfg = cfg.models.get("whisper", {}) if hasattr(cfg, "models") else {}
    model_name = asr_cfg.get("model", "base")
    device = asr_cfg.get("device", "cpu")
    stt = transcribe_tamil(str(p), model_name=model_name, device=device)
    transcript = clean_tamil_text(stt.get("text", ""))
    nlp_path = cfg.models.get("nlp_tamil", {}).get("path") if hasattr(cfg, "models") else None
    nlp_model = cfg.models.get("nlp_tamil", {}).get("model") if hasattr(cfg, "models") else None
    clf_name = nlp_path if nlp_path else (nlp_model or "ai4bharat/indic-bert")
    classifier = TamilCoercionClassifier(clf_name)
    coercion_prob, label = classifier.predict(transcript or " ")
    return {
        "transcript": transcript,
        "nlp_prob": float(coercion_prob),
        "label": label,
        "timestamps": [],
    }
