from typing import Dict, Any
from pathlib import Path
import time

from multimodal_coercion.engine.audio_utils import extract_audio_ffmpeg
from multimodal_coercion.speech.whisper_stt import transcribe_tamil
from multimodal_coercion.speech.text_preprocess import clean_tamil_text
from multimodal_coercion.facial_emotion.video_emotion_pipeline import run_video_emotion
from backend.models.nlp_intent import analyze_intent_score
from backend.models.voice_stress import voice_stress_score
from backend.scoring import combine_scores
from multimodal_coercion.core.config import project_root, get_config


# helper for logging timings

def _log_stage(name: str, start: float):
    now = time.time()
    print(f"[verify_video] {name} took {now - start:.2f}s")
    return now


def verify_video(video_path: str) -> Dict[str, Any]:
    base = project_root()
    cfg = get_config(base)

    # 1) Extract audio once and reuse for STT+stress
    t0 = time.time()
    tmp_wav = extract_audio_ffmpeg(video_path, sample_rate=cfg.default["audio"]["sample_rate"])
    t0 = _log_stage("audio extraction", t0)

    try:
        # 2) Whisper STT
        whisper_cfg = cfg.models.get("whisper", {})
        stt = transcribe_tamil(tmp_wav, model_name=whisper_cfg.get("model", "base"), device=whisper_cfg.get("device", "cpu"))
        t0 = _log_stage("speech-to-text", t0)
    finally:
        pass

    transcript = clean_tamil_text(stt.get("text", ""))
    t0 = _log_stage("text cleaning", t0)

    # 3) NLP Intent (uses proper HF model id internally)
    speech_intent, sentiment, coercion_flag = analyze_intent_score(transcript)
    t0 = _log_stage("nlp intent", t0)

    # 4) Facial emotion from video
    emo = run_video_emotion(video_path)
    t0 = _log_stage("video emotion", t0)
    fear = float(emo.get("avg_fear_prob", 0.0))
    stress = float(emo.get("avg_stress_prob", 0.0))
    emotion_score = max(0.0, 1.0 - max(fear, stress))

    # 5) Voice stress from audio (reuse same file)
    try:
        v_stress = voice_stress_score(tmp_wav)
    finally:
        pass
    t0 = _log_stage("voice stress", t0)

    # delete audio once at end
    try:
        Path(tmp_wav).unlink()
    except Exception:
        pass

    # 6) Combine scores to final willingness/confidence
    final_pct, final_label, action = combine_scores(speech_intent, emotion_score, v_stress)
    t0 = _log_stage("score combine", t0)

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

