from typing import Tuple


def combine_scores(speech_intent: float, emotion_stability: float, voice_stress: float) -> Tuple[float, str, str]:
    """
    Compute final willingness/confidence percentage and label using policy:
      - confidence < 40%  -> WORST
      - 40%–75%           -> AVERAGE
      - > 75%             -> GOOD
    Inputs:
      speech_intent: [0,1] higher is better
      emotion_stability: [0,1] higher is better
      voice_stress: [0,1] higher means more stress (worse)
    """
    voice_stability = max(0.0, 1.0 - float(voice_stress))
    confidence = 0.4 * float(speech_intent) + 0.3 * float(emotion_stability) + 0.3 * voice_stability
    pct = float(confidence) * 100.0
    if pct < 40.0:
        return pct, "WORST", "Restart full documentation verification"
    if pct <= 75.0:
        return pct, "AVERAGE", "Ask user to re-upload video"
    return pct, "GOOD", "Successfully Verified"

