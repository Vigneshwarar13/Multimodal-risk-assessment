from typing import Dict, Any
from pathlib import Path
import os
import tempfile

from multimodal_coercion.engine.audio_utils import extract_audio_ffmpeg
from multimodal_coercion.speech.whisper_stt import transcribe_tamil
from multimodal_coercion.speech.text_preprocess import clean_tamil_text
from multimodal_coercion.facial_emotion.video_emotion_pipeline import run_video_emotion
from backend.models.nlp_intent import analyze_intent_score
from backend.models.voice_stress import voice_stress_score
from backend.scoring import combine_scores
from multimodal_coercion.core.config import Config, project_root


def verify_video(video_path: str) -> Dict[str, Any]:
    base = project_root()
    cfg = Config(base)
    # 1) Extract audio
    tmp_wav = extract_audio_ffmpeg(video_path, sample_rate=cfg.default["audio"]["sample_rate"])
    try:
        # 2) Whisper STT
        whisper_cfg = cfg.models.get("whisper", {})
        stt = transcribe_tamil(tmp_wav, model_name=whisper_cfg.get("model", "base"), device=whisper_cfg.get("device", "cpu"))
    finally:
        try:
            Path(tmp_wav).unlink()
        except Exception:
            pass
    transcript = clean_tamil_text(stt.get("text", ""))

    # 3) NLP Intent (uses proper HF model id internally)
    speech_intent, sentiment, coercion_flag = analyze_intent_score(transcript)

    # 4) Facial emotion from video
    emo = run_video_emotion(video_path)
    fear = float(emo.get("avg_fear_prob", 0.0))
    stress = float(emo.get("avg_stress_prob", 0.0))
    emotion_score = max(0.0, 1.0 - max(fear, stress))

    # 5) Voice stress from audio
    tmp_wav2 = extract_audio_ffmpeg(video_path, sample_rate=cfg.default["audio"]["sample_rate"])
    try:
        v_stress = voice_stress_score(tmp_wav2)
    finally:
        try:
            Path(tmp_wav2).unlink()
        except Exception:
            pass

    # 6) Combine scores to final willingness/confidence
    final_pct, final_label, action = combine_scores(speech_intent, emotion_score, v_stress)

    em_sum = f"fear={fear:.2f}, stress={stress:.2f}, emotional_stability={emotion_score:.2f}"
    return {
        "transcript": transcript,
        "sentiment": sentiment,
        "coercion_detected": bool(coercion_flag),
        "willingness_score": int(round(final_pct)),
        "emotion_summary": em_sum,
        "emotion_score": float(emotion_score),
        "speech_intent_score": float(speech_intent),
        "voice_stress_score": float(v_stress),
        "final_decision": final_label,
        "recommended_action": action,
    }

