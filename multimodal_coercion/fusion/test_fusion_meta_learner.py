"""
Test suite for dynamic fusion weights meta-learner.

Tests the MetaLearnerFusion class and related functions.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path

from multimodal_coercion.fusion.fusion_model import (
    MetaLearnerFusion,
    FusionResult,
    train_meta_learner,
    load_and_fuse,
    fuse_scores,
    fuse_scores_with_confidence,
    classify_risk,
    fuse_and_classify,
    get_fusion_engine,
)


class TestFusionResult(unittest.TestCase):
    """Tests for FusionResult dataclass."""

    def test_result_creation(self):
        """Test creating a fusion result."""
        result = FusionResult(
            score=0.65,
            confidence=0.85,
            facial_weight=0.6,
            speech_weight=0.4,
            model_used="learned",
        )
        self.assertAlmostEqual(result.score, 0.65)
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertEqual(result.model_used, "learned")

    def test_weights_sum_to_one(self):
        """Test that weights approximately sum to 1."""
        result = FusionResult(
            score=0.50,
            confidence=0.50,
            facial_weight=0.6,
            speech_weight=0.4,
            model_used="static",
        )
        self.assertAlmostEqual(result.facial_weight + result.speech_weight, 1.0, places=5)


class TestMetaLearnerFusion(unittest.TestCase):
    """Tests for MetaLearnerFusion class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test meta-learner initialization."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))
        self.assertFalse(learner.is_trained)
        self.assertIsNone(learner.model)

    def test_fallback_to_static_fusion(self):
        """Test fallback to static 0.5/0.5 weights when not trained."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))
        result = learner.fuse(0.4, 0.6)

        self.assertAlmostEqual(result.score, 0.5)  # 0.5*0.4 + 0.5*0.6
        self.assertAlmostEqual(result.facial_weight, 0.5)
        self.assertAlmostEqual(result.speech_weight, 0.5)
        self.assertEqual(result.model_used, "static")

    def test_train_meta_learner(self):
        """Test training meta-learner on sample data."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))

        # Create synthetic training data
        facial_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.15, 0.85])
        speech_scores = np.array([0.25, 0.28, 0.75, 0.82, 0.18, 0.88])
        labels = np.array([0, 0, 1, 1, 0, 1])

        learner.train(facial_scores, speech_scores, labels)

        self.assertTrue(learner.is_trained)
        self.assertIsNotNone(learner.model)

    def test_train_requires_minimum_samples(self):
        """Test that training requires minimum samples."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))

        facial_scores = np.array([0.2, 0.3])
        speech_scores = np.array([0.25, 0.28])
        labels = np.array([0, 1])

        with self.assertRaises(ValueError):
            learner.train(facial_scores, speech_scores, labels)

    def test_train_requires_matched_lengths(self):
        """Test that training data must have matched lengths."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))

        facial_scores = np.array([0.2, 0.3, 0.7, 0.8])
        speech_scores = np.array([0.25, 0.28, 0.75])  # Length mismatch
        labels = np.array([0, 0, 1, 1])

        with self.assertRaises(ValueError):
            learner.train(facial_scores, speech_scores, labels)

    def test_model_persistence(self):
        """Test saving and loading trained model."""
        learner1 = MetaLearnerFusion(model_dir=str(self.model_dir))

        # Train model
        facial_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.15, 0.85])
        speech_scores = np.array([0.25, 0.28, 0.75, 0.82, 0.18, 0.88])
        labels = np.array([0, 0, 1, 1, 0, 1])

        learner1.train(facial_scores, speech_scores, labels)
        learner1.save_model()

        # Load in new instance
        learner2 = MetaLearnerFusion(model_dir=str(self.model_dir))
        loaded = learner2.load_model()

        self.assertTrue(loaded)
        self.assertTrue(learner2.is_trained)

        # Verify predictions are identical
        result1 = learner1.fuse(0.5, 0.5)
        result2 = learner2.fuse(0.5, 0.5)

        self.assertAlmostEqual(result1.score, result2.score, places=5)

    def test_fuse_learned_weights(self):
        """Test fusion with learned weights."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))

        # Train model
        facial_scores = np.array([0.1, 0.2, 0.7, 0.8, 0.15, 0.85])
        speech_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.85, 0.1])
        labels = np.array([1, 1, 0, 0, 1, 0])  # Opposite patterns

        learner.train(facial_scores, speech_scores, labels)

        # Test fusion
        result = learner.fuse(0.5, 0.5)

        self.assertTrue(learner.is_trained)
        self.assertEqual(result.model_used, "learned")
        self.assertGreater(result.confidence, 0.0)
        self.assertLess(result.confidence, 1.0)
        # Weights should sum to approximately 1
        self.assertAlmostEqual(
            result.facial_weight + result.speech_weight, 1.0, places=4
        )

    def test_score_clipping(self):
        """Test that scores and input are clipped to [0, 1]."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))
        result = learner.fuse(-0.5, 1.5)

        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_get_model_info(self):
        """Test getting model information."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))

        info = learner.get_model_info()
        self.assertFalse(info["is_trained"])

        # Train and check info
        facial_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.15, 0.85])
        speech_scores = np.array([0.25, 0.28, 0.75, 0.82, 0.18, 0.88])
        labels = np.array([0, 0, 1, 1, 0, 1])

        learner.train(facial_scores, speech_scores, labels)
        info = learner.get_model_info()

        self.assertTrue(info["is_trained"])
        self.assertIn("coefficients", info)
        self.assertIn("intercept", info)
        self.assertEqual(len(info["coefficients"]), 2)


class TestFusionFunctions(unittest.TestCase):
    """Tests for module-level fusion functions."""

    def test_classify_risk(self):
        """Test risk classification."""
        self.assertEqual(classify_risk(0.3), "Good")
        self.assertEqual(classify_risk(0.5), "Average")
        self.assertEqual(classify_risk(0.75), "Poor")

    def test_fuse_scores_basic(self):
        """Test basic score fusion (static fallback)."""
        score = fuse_scores(0.4, 0.6)
        self.assertAlmostEqual(score, 0.5)

    def test_fuse_scores_with_confidence(self):
        """Test score fusion with confidence output."""
        result = fuse_scores_with_confidence(0.6, 0.4)

        self.assertIsInstance(result, FusionResult)
        self.assertAlmostEqual(result.score, 0.5)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_fuse_and_classify(self):
        """Test fusing and classifying in one call."""
        score, label, result = fuse_and_classify(0.3, 0.3)

        self.assertAlmostEqual(score, 0.3)
        self.assertEqual(label, "Good")
        self.assertIsInstance(result, FusionResult)

    def test_get_fusion_engine_singleton(self):
        """Test that fusion engine is singleton."""
        engine1 = get_fusion_engine()
        engine2 = get_fusion_engine()

        self.assertIs(engine1, engine2)


class TestTrainMetaLearner(unittest.TestCase):
    """Tests for train_meta_learner function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_train_meta_learner_function(self):
        """Test train_meta_learner module function."""
        facial_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.15, 0.85])
        speech_scores = np.array([0.25, 0.28, 0.75, 0.82, 0.18, 0.88])
        labels = np.array([0, 0, 1, 1, 0, 1])

        summary = train_meta_learner(
            facial_scores, speech_scores, labels, model_dir=str(self.model_dir)
        )

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["n_samples"], 6)
        self.assertIn("facial_weight", summary)
        self.assertIn("speech_weight", summary)
        # Weights should sum to approximately 1
        total = summary["facial_weight"] + summary["speech_weight"]
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_model_file_created(self):
        """Test that model file is created on disk."""
        facial_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.15, 0.85])
        speech_scores = np.array([0.25, 0.28, 0.75, 0.82, 0.18, 0.88])
        labels = np.array([0, 0, 1, 1, 0, 1])

        train_meta_learner(
            facial_scores, speech_scores, labels, model_dir=str(self.model_dir)
        )

        model_path = self.model_dir / "meta_learner.pkl"
        self.assertTrue(model_path.exists())


class TestLoadAndFuse(unittest.TestCase):
    """Tests for load_and_fuse function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_load_and_fuse_with_trained_model(self):
        """Test load_and_fuse with trained model."""
        # Train model
        facial_scores = np.array([0.2, 0.3, 0.7, 0.8, 0.15, 0.85])
        speech_scores = np.array([0.25, 0.28, 0.75, 0.82, 0.18, 0.88])
        labels = np.array([0, 0, 1, 1, 0, 1])

        train_meta_learner(
            facial_scores, speech_scores, labels, model_dir=str(self.model_dir)
        )

        # Load and use in different process
        # (in real usage, this would be a different Python session)
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))
        loaded = learner.load_model()
        self.assertTrue(loaded)

        result = learner.fuse(0.5, 0.5)
        self.assertEqual(result.model_used, "learned")

    def test_load_and_fuse_fallback(self):
        """Test load_and_fuse falls back gracefully."""
        learner = MetaLearnerFusion(model_dir=str(self.model_dir))
        loaded = learner.load_model()
        self.assertFalse(loaded)  # Model doesn't exist

        # Should still work with static fusion
        result = learner.fuse(0.4, 0.6)
        self.assertEqual(result.model_used, "static")
        self.assertAlmostEqual(result.score, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
