import re
from typing import Tuple
from functools import lru_cache
import time
import torch
from transformers import AutoTokenizer, AutoModel


VOLUNTARY_PATTERNS = [
    r"\bsondha\b",
    r"\bvirupath[ao]du\b",
    r"\bconsent\b",
    r"\bwilling(ly)?\b",
    r"\bsale\s+pandr?en\b",
]
COERCION_PATTERNS = [
    r"\bzabardasti\b",
    r"\bpressure\b",
    r"\bthreat\b",
    r"\bforce[d]?\b",
]
HESITATION_PATTERNS = [
    r"\bmaybe\b",
    r"\bnot sure\b",
    r"\btheriyala\b",
    r"\bvenam\b",
    r"\billai\b",
]


def _pattern_score(text: str) -> Tuple[float, bool, bool]:
    t = text.lower()
    vol = any(re.search(p, t) for p in VOLUNTARY_PATTERNS)
    coer = any(re.search(p, t) for p in COERCION_PATTERNS)
    hes = any(re.search(p, t) for p in HESITATION_PATTERNS)
    score = 0.5
    if vol:
        score += 0.2
    if hes:
        score -= 0.15
    if coer:
        score -= 0.4
    score = max(0.0, min(1.0, score))
    return score, coer, hes


# simple cache for tokenizer/model pairs keyed by model_id
@lru_cache(maxsize=4)
def _load_intent_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device


def _log_stage(name: str, start: float):
    now = time.time()
    print(f"[nlp_intent] {name} took {now - start:.2f}s")
    return now


def detect_dialect_hint(text: str) -> str:
    """Return a dialect hint from text if recognizable."""
    t = (text or "").lower()
    if re.search(r"\b(லவ்|வீட்டுல|ஓஹோ)\b", t):
        return "Tamil (Southern/Dialectal)"
    if re.search(r"\b(அரு|அதில்|என்னோ)\b", t):
        return "Tamil (Standard)"
    return "Unknown"


def analyze_intent_score(text: str, model_id: str = "ai4bharat/indic-bert") -> Tuple[float, str, bool, float, bool, str]:
    """Compute willingness/coercion score + confidence + dialect fallback.

    Returns:
      score, sentiment, coercion_flag, nlp_confidence, dialect_fallback, dialect_hint
    """
    t0 = time.time()
    base_score, coer_flag, hes_flag = _pattern_score(text or "")
    t0 = _log_stage("pattern base_score", t0)

    dialect_hint = detect_dialect_hint(text)
    dialect_fallback = False
    nlp_confidence = 0.5

    try:
        tok, mdl, device = _load_intent_model(model_id)
        t0 = _log_stage("model load", t0)

        enc = tok(text or " ", truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            if hasattr(out, "last_hidden_state"):
                cls_vec = out.last_hidden_state[:, 0, :]  # [CLS]
                mag = float(torch.norm(cls_vec, dim=-1).mean().detach().cpu())
                nlp_confidence = float(1.0 / (1.0 + torch.exp(-torch.tensor(mag / 100.0)).item()))
            else:
                nlp_confidence = 0.5

            # If confidence low, fallback to pattern-based score
            if nlp_confidence < 0.55:
                score = base_score
                dialect_fallback = True
            else:
                score = 0.7 * base_score + 0.3 * nlp_confidence
        t0 = _log_stage("inference", t0)
    except Exception as e:
        score = base_score
        nlp_confidence = 0.0
        dialect_fallback = True
        t0 = _log_stage(f"inference failed: {e}", t0)

    score = max(0.0, min(1.0, float(score)))
    sentiment = "Positive" if score >= 0.6 else "Neutral" if score >= 0.4 else "Negative"
    return score, sentiment, bool(coer_flag), nlp_confidence, dialect_fallback, dialect_hint

