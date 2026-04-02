import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path

from multimodal_coercion.core.config import Config, project_root
from multimodal_coercion.core.logging import setup_logging
from multimodal_coercion.core.registry import ModelRegistry
from multimodal_coercion.core.persistence import Persistence
from multimodal_coercion.facial_emotion.pipeline import run_video_pipeline
from multimodal_coercion.speech.pipeline import run_speech_pipeline
from multimodal_coercion.fusion.fusion_model import fuse_features
from multimodal_coercion.risk.scoring import label_from_prob
from multimodal_coercion.calibration.baseline import BaselineCalibrator, BaselineResult


def run_full_pipeline(
    video_path: Optional[str],
    audio_path: Optional[str],
    base_dir: Optional[str] = None,
    baseline: Optional[BaselineResult] = None,
) -> Dict[str, Any]:
    """
    Run the full coercion detection pipeline on registration session.
    
    If baseline is provided, the final risk score will be normalized to be
    relative to the user's neutral baseline, reducing false positives.
    
    Args:
        video_path: Path to registration session video
        audio_path: Path to registration session audio
        base_dir: Base directory for configuration (defaults to project root)
        baseline: Optional BaselineResult from calibration phase
        
    Returns:
        Dict with session_id, probability (normalized if baseline provided),
        label, and individual modality features
    """
    logger = setup_logging()
    base = Path(base_dir) if base_dir else project_root()
    cfg = Config(base)
    reg = ModelRegistry(cfg.models, base)
    db = Persistence(cfg.default["persistence"]["db_path"], cfg.default["persistence"]["artifacts_dir"])

    video_feats = {}
    speech_feats = {}

    if video_path:
        video_feats = run_video_pipeline(video_path, enable_camera=False)
    if audio_path:
        speech_feats = run_speech_pipeline(audio_path)

    prob = fuse_features(video_feats, speech_feats)
    
    # Normalize score if baseline was provided
    if baseline is not None:
        logger.info(
            f"Applying baseline normalization. Original score: {prob:.3f}, "
            f"Baseline (facial: {baseline.facial_baseline:.3f}, "
            f"speech: {baseline.speech_baseline:.3f})"
        )
        calibrator = BaselineCalibrator(baseline.user_id)
        prob = calibrator.normalize_score(prob, baseline)
        logger.info(f"Normalized score: {prob:.3f}")
    
    t_good = float(cfg.thresholds["good_max"])
    t_poor = float(cfg.thresholds["poor_min"])
    label, p = label_from_prob(prob, t_good, t_poor)

    session_id = str(uuid.uuid4())
    meta = {
        "video_path": video_path,
        "audio_path": audio_path,
        "baseline_applied": baseline is not None,
    }
    if baseline:
        meta["baseline_focal_score"] = baseline.facial_baseline
        meta["baseline_speech_score"] = baseline.speech_baseline
    
    db.save_session(session_id, label, p, meta)

    result = {
        "session_id": session_id,
        "probability": float(p),
        "label": label,
        "video": video_feats,
        "speech": speech_feats,
        "baseline_applied": baseline is not None,
    }
    if baseline:
        result["baseline"] = {
            "facial_baseline": baseline.facial_baseline,
            "speech_baseline": baseline.speech_baseline,
        }
    
    logger.info(f"{label} {p:.3f}")
    return result


def run_calibration_phase(
    neutral_video_paths: List[str],
    neutral_audio_paths: List[str],
    user_id: str,
    base_dir: Optional[str] = None,
) -> BaselineResult:
    """
    Baseline calibration phase: analyze 3 neutral question responses.
    
    This should be called BEFORE the actual registration session analysis.
    The user is asked 3 neutral questions (e.g., "What is your full name?",
    "Where do you currently live?", "What is your occupation?"). Their
    facial expressions and speech patterns during these neutral responses
    establish their personal baseline for emotion and coercion indicators.
    
    Args:
        neutral_video_paths: List of 3 paths to neutral question videos
        neutral_audio_paths: List of 3 paths to neutral question audio clips
        user_id: Unique identifier for the user (session ID, phone, etc.)
        base_dir: Base directory for configuration (defaults to project root)
        
    Returns:
        BaselineResult containing calibrated baseline scores
        
    Raises:
        ValueError: If not exactly 3 samples provided
        
    Example:
        >>> baseline = run_calibration_phase(
        ...     ["neutral_q1.mp4", "neutral_q2.mp4", "neutral_q3.mp4"],
        ...     ["neutral_q1.wav", "neutral_q2.wav", "neutral_q3.wav"],
        ...     user_id="user_12345"
        ... )
        >>> print(f"Baseline facial: {baseline.facial_baseline:.3f}")
        >>> print(f"Baseline speech: {baseline.speech_baseline:.3f}")
    """
    import numpy as np
    
    if len(neutral_video_paths) != 3 or len(neutral_audio_paths) != 3:
        raise ValueError(
            f"Need exactly 3 neutral samples. "
            f"Got {len(neutral_video_paths)} videos and {len(neutral_audio_paths)} audios."
        )

    logger = setup_logging()
    logger.info(f"Starting baseline calibration for user {user_id}")

    base = Path(base_dir) if base_dir else project_root()
    cfg = Config(base)
    db = Persistence(
        cfg.default["persistence"]["db_path"],
        cfg.default["persistence"]["artifacts_dir"],
    )

    calibrator = BaselineCalibrator(user_id, persistence=db)

    # Process each neutral question response
    for idx, (video_path, audio_path) in enumerate(
        zip(neutral_video_paths, neutral_audio_paths), 1
    ):
        logger.info(f"Calibration question {idx}/3: analyzing response...")

        video_feats = run_video_pipeline(video_path, enable_camera=False) if video_path else {}
        speech_feats = run_speech_pipeline(audio_path) if audio_path else {}

        # Extract individual modality scores
        facial_score = 0.0
        if video_feats.get("emotion_probs"):
            facial_scores = np.array(video_feats["emotion_probs"], dtype=float)
            facial_score = float(np.clip(facial_scores.mean(), 0.0, 1.0))

        speech_score = 0.0
        if speech_feats.get("nlp_prob"):
            speech_score = float(np.clip(speech_feats["nlp_prob"], 0.0, 1.0))

        calibrator.record_sample(facial_score, speech_score)
        logger.info(
            f"  Q{idx} - Facial stress: {facial_score:.3f}, "
            f"Speech coercion: {speech_score:.3f}"
        )

    # Compute and save baseline
    baseline = calibrator.compute_baseline()
    logger.info(
        f"Baseline calibration complete for {user_id}. "
        f"Facial baseline: {baseline.facial_baseline:.3f}, "
        f"Speech baseline: {baseline.speech_baseline:.3f}"
    )

    return baseline


def run_full_pipeline_with_baseline(
    neutral_video_paths: List[str],
    neutral_audio_paths: List[str],
    session_video_path: Optional[str],
    session_audio_path: Optional[str],
    user_id: str,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete coercion detection pipeline: calibration + registration analysis.
    
    This is the recommended entry point for analyzing a registration session.
    It performs baseline calibration first (3 neutral questions), then analyzes
    the actual registration session using the personalized baseline to reduce
    false positives.
    
    Args:
        neutral_video_paths: List of 3 neutral question video paths
        neutral_audio_paths: List of 3 neutral question audio paths
        session_video_path: Path to registration session video
        session_audio_path: Path to registration session audio
        user_id: Unique identifier for the user
        base_dir: Base directory for configuration (defaults to project root)
        
    Returns:
        Dict with complete analysis results:
        - session_id: Unique session identifier
        - probability: Normalized risk score (0-1)
        - label: Risk category (Good/Average/Poor)
        - baseline: Computed baseline scores
        - baseline_applied: True (always applied in this function)
        - video: Video pipeline features
        - speech: Speech pipeline features
        
    Example:
        >>> result = run_full_pipeline_with_baseline(
        ...     neutral_videos, neutral_audios,
        ...     "registration.mp4", "registration.wav",
        ...     user_id="user_12345"
        ... )
        >>> print(f"Risk level: {result['label']}")
        >>> print(f"Confidence: {result['probability']:.1%}")
    """
    logger = setup_logging()

    # Phase 1: Baseline calibration
    logger.info("=" * 60)
    logger.info("PHASE 1: BASELINE CALIBRATION")
    logger.info("=" * 60)
    baseline = run_calibration_phase(
        neutral_video_paths, neutral_audio_paths, user_id, base_dir
    )

    # Phase 2: Registration session analysis with baseline normalization
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: REGISTRATION SESSION ANALYSIS")
    logger.info("=" * 60)
    result = run_full_pipeline(
        session_video_path, session_audio_path, base_dir, baseline=baseline
    )

    return result