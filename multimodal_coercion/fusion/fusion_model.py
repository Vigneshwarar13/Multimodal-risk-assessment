"""
Multimodal fusion system with optional meta-learner for dynamic weight optimization.

Supports two modes:
1. Static: Fixed 0.5/0.5 weights (backward compatible fallback)
2. Dynamic: Learned weights via LogisticRegression meta-learner
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

from multimodal_coercion.core.config import project_root


@dataclass
class FusionResult:
    """Result of multimodal fusion with confidence estimates."""
    score: float
    confidence: float
    facial_weight: float
    speech_weight: float
    model_used: str  # "static" or "learned"


class MetaLearnerFusion:
    """
    Dynamic fusion weights via meta-learner.
    
    Uses LogisticRegression to learn optimal weights for combining
    facial and speech coercion scores. Trained on labeled data,
    it predicts the probability of true coercion given the two
    modality scores.
    
    Provides graceful fallback to static 0.5/0.5 weights if
    trained model not available.
    """

    DEFAULT_MODEL_PATH = None  # Will be set in __init__

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize meta-learner fusion.
        
        Args:
            model_dir: Directory containing meta_learner.pkl. 
                      Defaults to models/fusion/
        """
        if model_dir is None:
            model_dir = Path(project_root()) / "models" / "fusion"
        
        self.model_dir = Path(model_dir)
        self.model: Optional[LogisticRegression] = None
        self.is_trained = False

    def load_model(self) -> bool:
        """
        Load trained meta-learner model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = self.model_dir / "meta_learner.pkl"
        
        if not model_path.exists():
            return False

        try:
            self.model = joblib.load(model_path)
            self.is_trained = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to load meta-learner model: {e}")
            self.is_trained = False
            return False

    def train(self, facial_scores: np.ndarray, speech_scores: np.ndarray, 
              labels: np.ndarray) -> None:
        """
        Train meta-learner on labeled data.
        
        Args:
            facial_scores: Array of facial coercion scores (0-1)
            speech_scores: Array of speech coercion scores (0-1)
            labels: Binary array of true labels (0=genuine, 1=coercion)
            
        Raises:
            ValueError: If data insufficient or mismatched shapes
        """
        facial_scores = np.asarray(facial_scores, dtype=float).flatten()
        speech_scores = np.asarray(speech_scores, dtype=float).flatten()
        labels = np.asarray(labels, dtype=int).flatten()

        if len(facial_scores) != len(speech_scores) or len(facial_scores) != len(labels):
            raise ValueError(
                f"Mismatched lengths: facial={len(facial_scores)}, "
                f"speech={len(speech_scores)}, labels={len(labels)}"
            )

        if len(facial_scores) < 5:
            raise ValueError(f"Need at least 5 samples for training, got {len(facial_scores)}")

        # Stack scores as features: shape (n_samples, 2)
        X = np.column_stack([facial_scores, speech_scores])

        # Train logistic regression (binary classifier)
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'  # Handle imbalanced data
        )
        self.model.fit(X, labels)
        self.is_trained = True

    def save_model(self) -> None:
        """Save trained model to disk."""
        if self.model is None or not self.is_trained:
            raise ValueError("No trained model to save")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "meta_learner.pkl"

        joblib.dump(self.model, model_path)

    def fuse(self, facial_score: float, speech_score: float) -> FusionResult:
        """
        Fuse facial and speech scores using trained model.
        
        If model not trained/loaded, falls back to static 0.5/0.5 weights.
        
        Args:
            facial_score: Facial coercion probability (0-1)
            speech_score: Speech coercion probability (0-1)
            
        Returns:
            FusionResult with fused score, confidence, and weights used
        """
        facial_score = float(np.clip(facial_score, 0.0, 1.0))
        speech_score = float(np.clip(speech_score, 0.0, 1.0))

        if not self.is_trained or self.model is None:
            # Fall back to static weights
            return self._fuse_static(facial_score, speech_score)

        return self._fuse_learned(facial_score, speech_score)

    def _fuse_learned(self, facial_score: float, speech_score: float) -> FusionResult:
        """Fuse using learned meta-learner model."""
        X = np.array([[facial_score, speech_score]])

        # Get prediction probability (probability of coercion)
        proba = self.model.predict_proba(X)[0]
        coercion_prob = float(proba[1])  # Probability of class 1 (coercion)
        confidence = float(np.max(proba))

        # Derive effective weights from model coefficients
        # Normalize coefficients to sum to 1
        weights = np.abs(self.model.coef_[0])
        weights = weights / (weights.sum() + 1e-8)
        facial_weight = float(weights[0])
        speech_weight = float(weights[1])

        return FusionResult(
            score=coercion_prob,
            confidence=confidence,
            facial_weight=facial_weight,
            speech_weight=speech_weight,
            model_used="learned"
        )

    def _fuse_static(self, facial_score: float, speech_score: float) -> FusionResult:
        """Fuse using static 0.5/0.5 weights (fallback)."""
        score = float(0.5 * facial_score + 0.5 * speech_score)
        return FusionResult(
            score=score,
            confidence=0.5,  # No confidence estimate in static mode
            facial_weight=0.5,
            speech_weight=0.5,
            model_used="static"
        )

    def _fuse_static_custom(
        self, facial_score: float, speech_score: float,
        facial_weight: float = 0.5, speech_weight: float = 0.5
    ) -> FusionResult:
        """
        Fuse using custom static weights.
        
        Used for confidence gating when transcription reliability is low.
        
        Args:
            facial_score: Facial coercion probability
            speech_score: Speech coercion probability
            facial_weight: Weight for facial (default 0.5)
            speech_weight: Weight for speech (default 0.5)
            
        Returns:
            FusionResult using custom weights
        """
        score = float(
            facial_weight * facial_score + speech_weight * speech_score
        )
        return FusionResult(
            score=score,
            confidence=0.5,  # No confidence estimate in static mode
            facial_weight=facial_weight,
            speech_weight=speech_weight,
            model_used="static"
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model.
        
        Returns:
            Dict with model metadata
        """
        info = {
            "is_trained": self.is_trained,
            "model_dir": str(self.model_dir),
            "model_path": str(self.model_dir / "meta_learner.pkl"),
            "model_file_exists": (self.model_dir / "meta_learner.pkl").exists(),
        }

        if self.is_trained and self.model is not None:
            info["coefficients"] = self.model.coef_[0].tolist()
            info["intercept"] = float(self.model.intercept_[0])
            info["classes"] = self.model.classes_.tolist()

        return info


# Global instance for convenient access
_meta_learner_instance = None


def get_fusion_engine(model_dir: Optional[str] = None) -> MetaLearnerFusion:
    """
    Get or create global meta-learner instance.
    
    Args:
        model_dir: Optional directory override for model location
        
    Returns:
        MetaLearnerFusion instance
    """
    global _meta_learner_instance
    if _meta_learner_instance is None:
        _meta_learner_instance = MetaLearnerFusion(model_dir)
        _meta_learner_instance.load_model()  # Try to load trained model
    return _meta_learner_instance


def fuse_features(video_features: Dict[str, Any], speech_features: Dict[str, Any]) -> float:
    """
    Fuse video and speech features into single coercion probability.
    
    Applies confidence gating: if speech transcription confidence is low,
    reduces speech weight to 0.2 and increases facial weight to 0.8.
    
    Uses meta-learner if available, falls back to static weights.
    
    Args:
        video_features: Dict from facial emotion pipeline
        speech_features: Dict from speech pipeline
        
    Returns:
        Fused coercion probability (0-1)
    """
    v = 0.0
    s = 0.0
    vp = video_features.get("emotion_probs")
    if isinstance(vp, list) and vp:
        arr = np.array(vp, dtype=float)
        v = float(np.clip(arr.mean(), 0.0, 1.0))
    sp = speech_features.get("nlp_prob")
    if isinstance(sp, (float, int)):
        s = float(np.clip(sp, 0.0, 1.0))
    
    # Check transcription confidence
    transcription_confidence = speech_features.get("transcription_confidence", 1.0)
    transcription_reliable = speech_features.get("transcription_reliable", True)
    
    # If transcription is not reliable, trust facial expression more
    engine = get_fusion_engine()
    if not transcription_reliable or transcription_confidence < 0.6:
        # Use confidence-gated static fusion: 0.8 facial + 0.2 speech
        result = engine._fuse_static_custom(v, s, facial_weight=0.8, speech_weight=0.2)
    else:
        # Use normal fusion (learned or static)
        result = engine.fuse(v, s)
    
    return result.score


def fuse_scores(facial_prob: float, speech_prob: float) -> float:
    """
    Fuse facial and speech scores using meta-learner.
    
    Uses trained meta-learner if available, otherwise static 0.5/0.5 weights.
    
    Args:
        facial_prob: Facial coercion probability (0-1)
        speech_prob: Speech coercion probability (0-1)
        
    Returns:
        Fused coercion probability (0-1)
    """
    engine = get_fusion_engine()
    result = engine.fuse(facial_prob, speech_prob)
    return result.score


def fuse_scores_with_confidence(facial_prob: float, speech_prob: float) -> FusionResult:
    """
    Fuse scores and return detailed result with confidence.
    
    Args:
        facial_prob: Facial coercion probability (0-1)
        speech_prob: Speech coercion probability (0-1)
        
    Returns:
        FusionResult with score, confidence, weights, and model info
    """
    engine = get_fusion_engine()
    return engine.fuse(facial_prob, speech_prob)


def classify_risk(score: float) -> str:
    """
    Classify risk level based on fused score.
    
    Args:
        score: Coercion probability (0-1)
        
    Returns:
        Risk category: "Good", "Average", or "Poor"
    """
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.4:
        return "Good"
    if score < 0.7:
        return "Average"
    return "Poor"


def fuse_and_classify(facial_prob: float, speech_prob: float) -> Tuple[float, str, FusionResult]:
    """
    Fuse scores and classify risk in one call.
    
    Args:
        facial_prob: Facial coercion probability (0-1)
        speech_prob: Speech coercion probability (0-1)
        
    Returns:
        Tuple of (score, label, FusionResult with details)
    """
    result = fuse_scores_with_confidence(facial_prob, speech_prob)
    label = classify_risk(result.score)
    return result.score, label, result


def train_meta_learner(
    facial_scores: np.ndarray,
    speech_scores: np.ndarray,
    labels: np.ndarray,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train meta-learner on labeled coercion detection data.
    
    Should be called during model development with historical data
    to learn optimal fusion weights.
    
    Args:
        facial_scores: Array of facial scores from genuine/coerced samples
        speech_scores: Array of speech scores from genuine/coerced samples
        labels: Array of ground truth labels (0=genuine, 1=coercion)
        model_dir: Optional directory for saving model
        
    Returns:
        Dict with training summary including model coefficients
        
    Example:
        >>> facial_scores = np.array([0.2, 0.3, 0.7, 0.8])
        >>> speech_scores = np.array([0.15, 0.25, 0.65, 0.75])
        >>> labels = np.array([0, 0, 1, 1])
        >>> summary = train_meta_learner(facial_scores, speech_scores, labels)
        >>> print(f"Model trained: facial_weight={summary['facial_weight']:.3f}")
    """
    engine = MetaLearnerFusion(model_dir)
    engine.train(facial_scores, speech_scores, labels)
    engine.save_model()

    info = engine.get_model_info()
    
    return {
        "status": "success",
        "n_samples": len(facial_scores),
        "model_path": str(engine.model_dir / "meta_learner.pkl"),
        "facial_weight": float(np.abs(engine.model.coef_[0][0])) /
                        (np.abs(engine.model.coef_[0]).sum() + 1e-8),
        "speech_weight": float(np.abs(engine.model.coef_[0][1])) /
                        (np.abs(engine.model.coef_[0]).sum() + 1e-8),
        "coefficients": info["coefficients"],
        "intercept": info["intercept"],
    }


def load_and_fuse(facial_score: float, speech_score: float) -> FusionResult:
    """
    Load meta-learner and fuse scores in one call.
    
    This is the primary inference function for using a trained meta-learner.
    Automatically falls back to static weights if model not found.
    
    Args:
        facial_score: Facial coercion probability (0-1)
        speech_score: Speech coercion probability (0-1)
        
    Returns:
        FusionResult with fused score and confidence
        
    Example:
        >>> result = load_and_fuse(0.35, 0.40)
        >>> print(f"Score: {result.score:.3f}, Confidence: {result.confidence:.2%}")
        >>> print(f"Used {result.model_used} fusion (facial={result.facial_weight:.2%})")
    """
    engine = MetaLearnerFusion()
    engine.load_model()
    return engine.fuse(facial_score, speech_score)
