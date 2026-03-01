import re
from typing import Tuple
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


def analyze_intent_score(text: str, model_id: str = "ai4bharat/indic-bert") -> Tuple[float, str, bool]:
    """
    Returns:
      - speech_intent_score in [0,1] (higher means more willing)
      - sentiment: 'Positive'/'Neutral'/'Negative'
      - coercion_flag: bool
    Uses proper HF loading syntax with a public model id.
    """
    base_score, coer_flag, hes_flag = _pattern_score(text or "")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
        enc = tok(text or " ", truncation=True, max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
            if hasattr(out, "last_hidden_state"):
                cls_vec = out.last_hidden_state[:, 0, :]  # [CLS]
                mag = float(torch.norm(cls_vec, dim=-1).mean().detach().cpu())
                conf = 1.0 / (1.0 + torch.exp(-torch.tensor(mag / 100.0))).item()
            else:
                conf = 0.5
        score = 0.7 * base_score + 0.3 * conf
    except Exception:
        score = base_score
    score = max(0.0, min(1.0, float(score)))
    sentiment = "Positive" if score >= 0.6 else "Neutral" if score >= 0.4 else "Negative"
    return score, sentiment, bool(coer_flag)

