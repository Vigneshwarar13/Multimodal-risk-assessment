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


def analyze_intent_score(text: str, model_id: str = "ai4bharat/indic-bert") -> Tuple[float, str, bool]:
    """Compute willingness/coercion score using lightweight heuristics + HF model.

    The expensive transformer is loaded only once per ``model_id`` via a
    simple LRU cache. We also emit timing logs that can be observed in the
    server console.
    """
    t0 = time.time()
    base_score, coer_flag, hes_flag = _pattern_score(text or "")
    t0 = _log_stage("pattern base_score", t0)

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
                conf = 1.0 / (1.0 + torch.exp(-torch.tensor(mag / 100.0))).item()
            else:
                conf = 0.5
        score = 0.7 * base_score + 0.3 * conf
        t0 = _log_stage("inference", t0)
    except Exception:
        score = base_score
        t0 = _log_stage("inference failed", t0)

    score = max(0.0, min(1.0, float(score)))
    sentiment = "Positive" if score >= 0.6 else "Neutral" if score >= 0.4 else "Negative"
    return score, sentiment, bool(coer_flag)

