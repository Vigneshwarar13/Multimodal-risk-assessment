"""
Testing guide for TASK 1: Baseline Calibration

This module provides unit tests and examples for the baseline calibration system.
"""

import unittest
import tempfile
import json
from pathlib import Path
from multimodal_coercion.calibration.baseline import BaselineCalibrator, BaselineResult


class TestBaselineResult(unittest.TestCase):
    """Tests for BaselineResult dataclass."""

    def test_baseline_result_creation(self):
        """Test creating a baseline result."""
        result = BaselineResult(
            user_id="user_123",
            facial_baseline=0.3,
            speech_baseline=0.25,
            sample_count=3,
            timestamp="2024-04-02T10:00:00",
        )
        self.assertEqual(result.user_id, "user_123")
        self.assertAlmostEqual(result.facial_baseline, 0.3)
        self.assertAlmostEqual(result.speech_baseline, 0.25)

    def test_baseline_to_dict(self):
        """Test converting baseline to dictionary."""
        result = BaselineResult(
            user_id="user_123",
            facial_baseline=0.3,
            speech_baseline=0.25,
            sample_count=3,
            timestamp="2024-04-02T10:00:00",
        )
        d = result.to_dict()
        self.assertEqual(d["user_id"], "user_123")
        self.assertEqual(d["facial_baseline"], 0.3)
        self.assertIsInstance(d["neutral_responses"], list)

    def test_baseline_from_dict(self):
        """Test creating baseline from dictionary."""
        d = {
            "user_id": "user_123",
            "facial_baseline": 0.3,
            "speech_baseline": 0.25,
            "sample_count": 3,
            "timestamp": "2024-04-02T10:00:00",
            "neutral_responses": [],
        }
        result = BaselineResult.from_dict(d)
        self.assertEqual(result.user_id, "user_123")
        self.assertAlmostEqual(result.facial_baseline, 0.3)


class TestBaselineCalibrator(unittest.TestCase):
    """Tests for BaselineCalibrator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calibrator = BaselineCalibrator(user_id="test_user_001")

    def test_record_samples(self):
        """Test recording calibration samples."""
        self.calibrator.record_sample(0.2, 0.15)
        self.calibrator.record_sample(0.25, 0.18)
        self.calibrator.record_sample(0.22, 0.16)

        self.assertEqual(len(self.calibrator.samples), 3)
        self.assertAlmostEqual(self.calibrator.samples[0]["facial_score"], 0.2)
        self.assertAlmostEqual(self.calibrator.samples[0]["speech_score"], 0.15)

    def test_clip_scores_to_valid_range(self):
        """Test that scores are clipped to [0, 1]."""
        self.calibrator.record_sample(-0.5, 1.5)
        sample = self.calibrator.samples[0]
        self.assertAlmostEqual(sample["facial_score"], 0.0)
        self.assertAlmostEqual(sample["speech_score"], 1.0)

    def test_compute_baseline_requires_three_samples(self):
        """Test that compute_baseline requires exactly 3 samples."""
        self.calibrator.record_sample(0.2, 0.15)
        self.calibrator.record_sample(0.25, 0.18)

        with self.assertRaises(ValueError):
            self.calibrator.compute_baseline()

    def test_compute_baseline_success(self):
        """Test successful baseline computation."""
        self.calibrator.record_sample(0.2, 0.15)
        self.calibrator.record_sample(0.25, 0.18)
        self.calibrator.record_sample(0.22, 0.16)

        baseline = self.calibrator.compute_baseline()

        # Average of [0.2, 0.25, 0.22]
        expected_facial = (0.2 + 0.25 + 0.22) / 3.0
        # Average of [0.15, 0.18, 0.16]
        expected_speech = (0.15 + 0.18 + 0.16) / 3.0

        self.assertAlmostEqual(baseline.facial_baseline, expected_facial, places=4)
        self.assertAlmostEqual(baseline.speech_baseline, expected_speech, places=4)
        self.assertEqual(baseline.sample_count, 3)
        self.assertEqual(baseline.user_id, "test_user_001")

    def test_normalize_score_with_no_baseline(self):
        """Test normalization returns original score when no baseline."""
        score = self.calibrator.normalize_score(0.6)
        self.assertAlmostEqual(score, 0.6)

    def test_normalize_score_score_equals_baseline(self):
        """Test normalization when session equals baseline (no deviation)."""
        baseline = BaselineResult(
            user_id="test_user",
            facial_baseline=0.4,
            speech_baseline=0.4,
            sample_count=3,
            timestamp="2024-04-02",
        )
        self.calibrator.baseline = baseline

        # Average baseline = 0.4
        # Normalized(0.4) = (0.4 - 0.4) / (1 - 0.4) = 0
        normalized = self.calibrator.normalize_score(0.4, baseline)
        self.assertAlmostEqual(normalized, 0.0, places=4)

    def test_normalize_score_score_at_max(self):
        """Test normalization when session score is at max (1.0)."""
        baseline = BaselineResult(
            user_id="test_user",
            facial_baseline=0.3,
            speech_baseline=0.3,
            sample_count=3,
            timestamp="2024-04-02",
        )
        self.calibrator.baseline = baseline

        # Average baseline = 0.3
        # Normalized(1.0) = (1.0 - 0.3) / (1 - 0.3) = 0.7 / 0.7 = 1.0
        normalized = self.calibrator.normalize_score(1.0, baseline)
        self.assertAlmostEqual(normalized, 1.0, places=4)

    def test_normalize_score_intermediate(self):
        """Test normalization with intermediate score."""
        baseline = BaselineResult(
            user_id="test_user",
            facial_baseline=0.2,
            speech_baseline=0.2,
            sample_count=3,
            timestamp="2024-04-02",
        )
        self.calibrator.baseline = baseline

        # Average baseline = 0.2
        # Normalized(0.6) = (0.6 - 0.2) / (1 - 0.2) = 0.4 / 0.8 = 0.5
        normalized = self.calibrator.normalize_score(0.6, baseline)
        self.assertAlmostEqual(normalized, 0.5, places=4)

    def test_normalize_multiple_scores(self):
        """Test utility function for normalizing multiple scores."""
        baseline = BaselineResult(
            user_id="test_user",
            facial_baseline=0.3,
            speech_baseline=0.3,
            sample_count=3,
            timestamp="2024-04-02",
        )

        scores = [0.3, 0.5, 0.7, 1.0]
        normalized = BaselineCalibrator.normalize_multiple_scores(scores, baseline)

        self.assertEqual(len(normalized), 4)
        self.assertAlmostEqual(normalized[0], 0.0, places=4)  # 0.3 (no deviation)
        # All values should be in [0, 1]
        for score in normalized:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestBaselineIntegration(unittest.TestCase):
    """Integration tests for baseline with persistence."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_manual_usage_example(self):
        """
        Example of manual usage without persistence.
        
        This shows the typical workflow:
        1. Calibration phase: record 3 neutral responses
        2. Compute baseline
        3. Use baseline to normalize registration session score
        """
        # Step 1: Calibration
        calibrator = BaselineCalibrator(user_id="subject_001")

        # Simulate 3 neutral question responses
        # (In real usage, these come from video/speech analysis)
        calibrator.record_sample(facial_score=0.15, speech_score=0.12)
        calibrator.record_sample(facial_score=0.18, speech_score=0.14)
        calibrator.record_sample(facial_score=0.17, speech_score=0.13)

        baseline = calibrator.compute_baseline()

        print(f"\nBaseline for {baseline.user_id}:")
        print(f"  Facial baseline: {baseline.facial_baseline:.3f}")
        print(f"  Speech baseline: {baseline.speech_baseline:.3f}")

        # Step 2: Registration session analysis
        # (In real usage, this comes from actual session analysis)
        session_scores = [
            0.35,  # Fairly stressed
            0.55,  # More stressed
            0.75,  # Very stressed
        ]

        print(f"\nSession scores (normalized):")
        for i, raw_score in enumerate(session_scores, 1):
            norm_score = calibrator.normalize_score(raw_score, baseline)
            print(f"  Response {i}: raw={raw_score:.2f} -> normalized={norm_score:.3f}")

        # Verify normalization properties
        # - Score below baseline should normalize to ~0
        low_score = calibrator.normalize_score(baseline.facial_baseline - 0.01, baseline)
        self.assertLess(low_score, 0.2)

        # - Score at average baseline should normalize close to 0 (within floating point precision)
        avg_baseline = (baseline.facial_baseline + baseline.speech_baseline) / 2.0
        baseline_score = calibrator.normalize_score(avg_baseline, baseline)
        self.assertAlmostEqual(baseline_score, 0.0, places=2)


def print_example_usage():
    """Print example usage patterns."""
    print("\n" + "=" * 70)
    print("BASELINE CALIBRATION - USAGE EXAMPLES")
    print("=" * 70)

    print("\n1. BASIC USAGE (in-memory only):")
    print("""
from multimodal_coercion.calibration.baseline import BaselineCalibrator

# Create calibrator
calibrator = BaselineCalibrator(user_id="user_12345")

# Record 3 neutral responses (from video/speech analysis)
calibrator.record_sample(facial_score=0.20, speech_score=0.18)
calibrator.record_sample(facial_score=0.22, speech_score=0.19)
calibrator.record_sample(facial_score=0.21, speech_score=0.17)

# Compute baseline
baseline = calibrator.compute_baseline()
print(f"Baseline: facial={baseline.facial_baseline:.3f}, "
      f"speech={baseline.speech_baseline:.3f}")

# Normalize session score
session_score = 0.65
normalized = calibrator.normalize_score(session_score, baseline)
print(f"Session score {session_score:.2f} -> normalized: {normalized:.3f}")
    """)

    print("\n2. WITH PERSISTENCE (recommended for production):")
    print("""
from multimodal_coercion.core.persistence import Persistence
from multimodal_coercion.calibration.baseline import BaselineCalibrator

# Initialize persistence layer
db = Persistence(
    db_path="./data.db",
    artifacts_dir="./artifacts"
)

# Create calibrator with persistence
calibrator = BaselineCalibrator(user_id="user_12345", persistence=db)

# Record samples (automatically saved to artifacts/baselines/user_12345_baseline.json)
calibrator.record_sample(0.20, 0.18)
calibrator.record_sample(0.22, 0.19)
calibrator.record_sample(0.21, 0.17)

baseline = calibrator.compute_baseline()  # Saved automatically

# Later: load baseline for same user
calibrator2 = BaselineCalibrator(user_id="user_12345", persistence=db)
loaded_baseline = calibrator2.load_baseline()
    """)

    print("\n3. USING FULL PIPELINE WITH BASELINE:")
    print("""
from multimodal_coercion.orchestrator.run_pipeline import (
    run_calibration_phase,
    run_full_pipeline_with_baseline,
)

# Option A: Separate steps
baseline = run_calibration_phase(
    neutral_video_paths=["q1.mp4", "q2.mp4", "q3.mp4"],
    neutral_audio_paths=["q1.wav", "q2.wav", "q3.wav"],
    user_id="user_12345"
)

result = run_full_pipeline(
    video_path="registration.mp4",
    audio_path="registration.wav",
    baseline=baseline  # Apply normalization
)

# Option B: All-in-one
result = run_full_pipeline_with_baseline(
    neutral_video_paths=["q1.mp4", "q2.mp4", "q3.mp4"],
    neutral_audio_paths=["q1.wav", "q2.wav", "q3.wav"],
    session_video_path="registration.mp4",
    session_audio_path="registration.wav",
    user_id="user_12345"
)

print(f"Risk: {result['label']} ({result['probability']:.1%})")
    """)

    print("\n4. BATCH NORMALIZATION:")
    print("""
from multimodal_coercion.calibration.baseline import BaselineCalibrator, BaselineResult

baseline = BaselineResult(
    user_id="user_123",
    facial_baseline=0.25,
    speech_baseline=0.20,
    sample_count=3,
    timestamp="2024-04-02T10:00:00"
)

scores = [0.35, 0.45, 0.55, 0.65, 0.75]
normalized = BaselineCalibrator.normalize_multiple_scores(scores, baseline)

for raw, norm in zip(scores, normalized):
    print(f"{raw:.2f} -> {norm:.3f}")
    """)


if __name__ == "__main__":
    # Run unit tests
    print("\nRunning unit tests for baseline calibration...\n")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Print usage examples
    print_example_usage()
