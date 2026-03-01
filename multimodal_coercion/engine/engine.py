from typing import Dict, Any
from pathlib import Path
import os
import tempfile

from backend.unified_engine import verify_video as unified_verify_video


def decide_final(coercion: bool, fear: float, sentiment: str, score: int) -> Dict[str, str]:
    if coercion or fear > 0.4:
        return {
            "final_decision": "Worst",
            "recommended_action": "Restart verification and re-check documentation from beginning",
        }
    if sentiment in {"Neutral", "Negative"} or score < 50:
        return {"final_decision": "Average", "recommended_action": "Reupload video"}
    return {"final_decision": "Good", "recommended_action": "Successfully verified"}


def verify_video(video_path: str) -> Dict[str, Any]:
    return unified_verify_video(video_path)
