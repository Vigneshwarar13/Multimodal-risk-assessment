"""
Baseline calibration system to reduce false positives in coercion detection.

A user's emotional baseline is established by analyzing 3 neutral question responses
before analyzing the registration session. The baseline scores are then used to
normalize the final risk score, making it relative to the individual's natural
emotional response pattern.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import numpy as np


@dataclass
class BaselineResult:
    """
    Stores baseline calibration data for a user session.
    
    Attributes:
        user_id: Unique identifier for the user (typically a session ID)
        facial_baseline: Average facial stress score from neutral questions (0-1)
        speech_baseline: Average speech coercion score from neutral questions (0-1)
        sample_count: Number of neutral responses recorded (should be 3)
        timestamp: ISO format timestamp when baseline was computed
        neutral_responses: List of raw calibration measurements for audit trail
    """
    user_id: str
    facial_baseline: float
    speech_baseline: float
    sample_count: int
    timestamp: str
    neutral_responses: List[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        data = asdict(self)
        if data.get("neutral_responses") is None:
            data["neutral_responses"] = []
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineResult":
        """Create instance from JSON-serializable dictionary."""
        return cls(**data)


class BaselineCalibrator:
    """
    Manages baseline calibration for multimodal coercion detection.
    
    Usage:
        1. Create instance: calibrator = BaselineCalibrator(user_id, persistence_instance)
        2. Record 3 neutral responses: calibrator.record_sample(facial_score, speech_score)
        3. Compute baseline: baseline = calibrator.compute_baseline()
        4. Use for normalization: normalized = calibrator.normalize_score(session_score, baseline)
    """

    def __init__(self, user_id: str, persistence=None):
        """
        Initialize calibrator for a user.
        
        Args:
            user_id: Unique identifier for the user (session ID, phone number, etc.)
            persistence: Optional Persistence instance for saving/loading baselines.
                        If None, baselines are stored in memory only.
        """
        self.user_id = user_id
        self.persistence = persistence
        self.samples: List[Dict[str, float]] = []
        self.baseline: Optional[BaselineResult] = None

    def record_sample(self, facial_score: float, speech_score: float) -> None:
        """
        Record a calibration sample from a neutral response.
        
        Call this method 3 times for neutral questions before proceeding to
        the actual registration session analysis.
        
        Args:
            facial_score: Facial stress probability from neutral question (0-1)
            speech_score: Speech coercion probability from neutral question (0-1)
        """
        facial_score = float(np.clip(facial_score, 0.0, 1.0))
        speech_score = float(np.clip(speech_score, 0.0, 1.0))

        sample = {
            "facial_score": facial_score,
            "speech_score": speech_score,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.samples.append(sample)

    def compute_baseline(self) -> BaselineResult:
        """
        Compute baseline from recorded samples.
        
        Requires exactly 3 samples. Averages facial and speech scores
        to establish the user's neutral emotion profile.
        
        Returns:
            BaselineResult with computed baseline values
            
        Raises:
            ValueError: If fewer than 3 samples recorded
        """
        if len(self.samples) < 3:
            raise ValueError(
                f"Need 3 neutral samples for baseline, got {len(self.samples)}"
            )

        facial_scores = [s["facial_score"] for s in self.samples[:3]]
        speech_scores = [s["speech_score"] for s in self.samples[:3]]

        facial_baseline = float(np.mean(facial_scores))
        speech_baseline = float(np.mean(speech_scores))

        self.baseline = BaselineResult(
            user_id=self.user_id,
            facial_baseline=facial_baseline,
            speech_baseline=speech_baseline,
            sample_count=3,
            timestamp=datetime.utcnow().isoformat(),
            neutral_responses=self.samples[:3],
        )

        # Persist if persistence layer available
        if self.persistence is not None:
            self._save_baseline(self.baseline)

        return self.baseline

    def load_baseline(self) -> Optional[BaselineResult]:
        """
        Load previously computed baseline for this user.
        
        Returns:
            BaselineResult if found, None otherwise
        """
        if self.persistence is not None:
            baseline_data = self._get_baseline_from_persistence()
            if baseline_data:
                self.baseline = BaselineResult.from_dict(baseline_data)
                return self.baseline
        return None

    def normalize_score(
        self, session_score: float, baseline: Optional[BaselineResult] = None
    ) -> float:
        """
        Normalize session risk score to be relative to baseline.
        
        Formula: normalized = (session_score - baseline_score) / (1 - baseline_score + eps)
        
        This makes the score relative to the individual's natural emotional baseline,
        reducing false positives for naturally expressive or tense individuals.
        
        Args:
            session_score: Fused coercion probability from registration session (0-1)
            baseline: BaselineResult to use. If None, uses self.baseline.
            
        Returns:
            Normalized score (typically 0-1 but can exceed 1.0 for extreme deviations)
        """
        if baseline is None:
            baseline = self.baseline

        if baseline is None:
            # No baseline available, return original score
            return float(np.clip(session_score, 0.0, 1.0))

        session_score = float(np.clip(session_score, 0.0, 1.0))

        # Average the baseline scores (equal weight to facial and speech)
        avg_baseline = (baseline.facial_baseline + baseline.speech_baseline) / 2.0

        # Normalize: scale the deviation from baseline to [0, 1] range
        # If session_score == avg_baseline, normalized = 0 (no deviation)
        # If session_score == 1.0 and baseline == 0, normalized = 1.0
        eps = 1e-6
        normalized = (session_score - avg_baseline) / (1.0 - avg_baseline + eps)

        # Clip to [0, 1] for risk classification
        return float(np.clip(normalized, 0.0, 1.0))

    def _save_baseline(self, baseline: BaselineResult) -> None:
        """
        Save baseline to persistence layer.
        
        Stores as JSON in artifacts directory with user_id as filename.
        """
        if self.persistence is None:
            return

        artifacts_dir = self.persistence.artifacts_dir
        baseline_dir = Path(artifacts_dir) / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        baseline_file = baseline_dir / f"{self.user_id}_baseline.json"
        with baseline_file.open("w", encoding="utf-8") as f:
            json.dump(baseline.to_dict(), f, indent=2)

    def _get_baseline_from_persistence(self) -> Optional[Dict[str, Any]]:
        """
        Load baseline from persistence layer.
        
        Returns:
            Dict with baseline data if found, None otherwise
        """
        if self.persistence is None:
            return None

        artifacts_dir = self.persistence.artifacts_dir
        baseline_file = Path(artifacts_dir) / "baselines" / f"{self.user_id}_baseline.json"

        if baseline_file.exists():
            with baseline_file.open("r", encoding="utf-8") as f:
                return json.load(f)

        return None

    @staticmethod
    def normalize_multiple_scores(
        session_scores: List[float], baseline: BaselineResult
    ) -> List[float]:
        """
        Utility to normalize multiple session scores using same baseline.
        
        Args:
            session_scores: List of risk scores to normalize
            baseline: BaselineResult to apply
            
        Returns:
            List of normalized scores
        """
        calibrator = BaselineCalibrator(baseline.user_id)
        calibrator.baseline = baseline
        return [calibrator.normalize_score(score, baseline) for score in session_scores]
