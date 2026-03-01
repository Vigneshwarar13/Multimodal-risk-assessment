import uuid
from typing import Optional, Dict, Any
from pathlib import Path

from multimodal_coercion.core.config import Config, project_root
from multimodal_coercion.core.logging import setup_logging
from multimodal_coercion.core.registry import ModelRegistry
from multimodal_coercion.core.persistence import Persistence
from multimodal_coercion.facial_emotion.pipeline import run_video_pipeline
from multimodal_coercion.speech.pipeline import run_speech_pipeline
from multimodal_coercion.fusion.fusion_model import fuse_features
from multimodal_coercion.risk.scoring import label_from_prob


def run_full_pipeline(
    video_path: Optional[str],
    audio_path: Optional[str],
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
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
    t_good = float(cfg.thresholds["good_max"])
    t_poor = float(cfg.thresholds["poor_min"])
    label, p = label_from_prob(prob, t_good, t_poor)

    session_id = str(uuid.uuid4())
    meta = {"video_path": video_path, "audio_path": audio_path}
    db.save_session(session_id, label, p, meta)

    result = {
        "session_id": session_id,
        "probability": float(p),
        "label": label,
        "video": video_feats,
        "speech": speech_feats,
    }
    logger.info(f"{label} {p:.3f}")
    return result

