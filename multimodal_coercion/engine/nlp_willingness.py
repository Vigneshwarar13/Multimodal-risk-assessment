import re
from typing import Tuple


VOLUNTARY_PATTERNS = [
    r"\bvirup[aā]th[ao]du\b",
    r"\bsondha\b",
    r"\bconsent\b",
    r"\bwill\b",
]
COERCION_PATTERNS = [
    r"\bzab[aā]rdasti\b",
    r"\bpressure\b",
    r"\bthreat\b",
    r"\bforce\b",
]


def analyze_transcript(text: str, coerce_prob: float, pred_label: str) -> Tuple[str, bool, int]:
    t = text.lower()
    vol = any(re.search(p, t) for p in VOLUNTARY_PATTERNS)
    coer = any(re.search(p, t) for p in COERCION_PATTERNS) or pred_label.lower() == "coercion"
    base = max(0.0, min(1.0, 1.0 - float(coerce_prob)))
    score = int(round(base * 100))
    if vol:
        score = min(100, score + 10)
    if coer:
        score = max(0, score - 40)
    sentiment = "Positive" if score >= 60 else "Neutral" if score >= 40 else "Negative"
    return sentiment, coer, score

