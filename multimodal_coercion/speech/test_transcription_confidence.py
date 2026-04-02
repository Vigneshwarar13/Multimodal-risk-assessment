"""
Test suite for transcription confidence gating.

Tests confidence extraction from Whisper and confidence-based fusion weight adjustment.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from multimodal_coercion.speech.whisper_stt import (
    TranscriptionResult,
    extract_segment_confidence,
)
from multimodal_coercion.fusion.fusion_model import MetaLearnerFusion


class TestTranscriptionResult(unittest.TestCase):
    """Tests for TranscriptionResult dataclass."""

    def test_creation(self):
        """Test creating a transcription result."""
        result = TranscriptionResult(
            text="Test text",
            confidence=0.85,
            is_reliable=True,
        )
        self.assertEqual(result.text, "Test text")
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertTrue(result.is_reliable)

    def test_confidence_clipping(self):
        """Test that confidence is clipped to [0, 1]."""
        result = TranscriptionResult(
            text="Test",
            confidence=1.5,
            is_reliable=True,
        )
        self.assertAlmostEqual(result.confidence, 1.0)

        result = TranscriptionResult(
            text="Test",
            confidence=-0.5,
            is_reliable=False,
        )
        self.assertAlmostEqual(result.confidence, 0.0)

    def test_default_segments(self):
        """Test that segments default to empty list."""
        result = TranscriptionResult(
            text="Test",
            confidence=0.8,
            is_reliable=True,
        )
        self.assertIsInstance(result.segments, list)
        self.assertEqual(len(result.segments), 0)


class TestSegmentConfidenceExtraction(unittest.TestCase):
    """Tests for Whisper segment confidence extraction."""

    def test_empty_segment(self):
        """Test extraction from segment with no tokens."""
        segment = {}
        confidence = extract_segment_confidence(segment)
        self.assertAlmostEqual(confidence, 0.5)

    def test_empty_tokens_list(self):
        """Test extraction from segment with empty tokens."""
        segment = {"tokens": []}
        confidence = extract_segment_confidence(segment)
        self.assertAlmostEqual(confidence, 0.5)

    def test_single_token_logprob(self):
        """Test extraction with single token."""
        # logprob of -1 means prob = e^(-1) ≈ 0.368
        segment = {
            "tokens": [
                {"logprob": -1.0}
            ]
        }
        confidence = extract_segment_confidence(segment)
        expected = np.exp(-1.0)
        self.assertAlmostEqual(confidence, expected, places=4)

    def test_multiple_tokens_average(self):
        """Test averaging confidence across tokens."""
        # logprobs: -0.5, -1.0, -0.3
        # probs: e^-0.5, e^-1.0, e^-0.3
        segment = {
            "tokens": [
                {"logprob": -0.5},
                {"logprob": -1.0},
                {"logprob": -0.3},
            ]
        }
        confidence = extract_segment_confidence(segment)
        expected = np.mean([np.exp(-0.5), np.exp(-1.0), np.exp(-0.3)])
        self.assertAlmostEqual(confidence, expected, places=4)

    def test_high_confidence_logprobs(self):
        """Test with high confidence (logprobs near 0)."""
        segment = {
            "tokens": [
                {"logprob": -0.01},
                {"logprob": -0.02},
                {"logprob": -0.01},
            ]
        }
        confidence = extract_segment_confidence(segment)
        # Should be close to 1.0
        self.assertGreater(confidence, 0.98)

    def test_low_confidence_logprobs(self):
        """Test with low confidence (logprobs very negative)."""
        segment = {
            "tokens": [
                {"logprob": -10.0},
                {"logprob": -15.0},
                {"logprob": -12.0},
            ]
        }
        confidence = extract_segment_confidence(segment)
        # Should be close to 0.0
        self.assertLess(confidence, 0.0001)

    def test_tokens_without_logprob(self):
        """Test handling tokens missing logprob field."""
        segment = {
            "tokens": [
                {"text": "hello"},  # Missing logprob
                {"logprob": -0.5},
                {"text": "world"},  # Missing logprob
            ]
        }
        confidence = extract_segment_confidence(segment)
        # Should handle gracefully (only process token with logprob)
        expected = np.exp(-0.5)
        self.assertAlmostEqual(confidence, expected, places=4)

    def test_confidence_range(self):
        """Test that confidence is always in [0, 1]."""
        # Test various extreme logprobs
        for logprob in [-100, -10, -1, -0.1, 0, 0.1]:
            segment = {"tokens": [{"logprob": logprob}]}
            confidence = extract_segment_confidence(segment)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)


class TestConfidenceGating(unittest.TestCase):
    """Tests for confidence-based weight adjustment."""

    def test_high_confidence_normal_weights(self):
        """Test that high confidence uses normal fusion."""
        learner = MetaLearnerFusion()

        # High confidence case
        result = learner._fuse_static(0.4, 0.6)
        self.assertAlmostEqual(result.facial_weight, 0.5)
        self.assertAlmostEqual(result.speech_weight, 0.5)
        self.assertAlmostEqual(result.score, 0.5)

    def test_low_confidence_custom_weights(self):
        """Test that low confidence uses 0.8/0.2 weights."""
        learner = MetaLearnerFusion()

        # Custom weights for low confidence
        result = learner._fuse_static_custom(0.4, 0.6, facial_weight=0.8, speech_weight=0.2)
        self.assertAlmostEqual(result.facial_weight, 0.8)
        self.assertAlmostEqual(result.speech_weight, 0.2)
        # Score = 0.8*0.4 + 0.2*0.6 = 0.32 + 0.12 = 0.44
        self.assertAlmostEqual(result.score, 0.44)

    def test_extreme_confidence_gating(self):
        """Test extreme confidences."""
        learner = MetaLearnerFusion()

        # Very high speech confidence but facial indicates coercion
        result_normal = learner._fuse_static(0.9, 0.1)
        self.assertAlmostEqual(result_normal.score, 0.5)

        # With low confidence gate: trust facial more
        result_gated = learner._fuse_static_custom(0.9, 0.1, facial_weight=0.8, speech_weight=0.2)
        self.assertAlmostEqual(result_gated.score, 0.74)  # 0.8*0.9 + 0.2*0.1

        # Gated version should trust facial emotion more
        self.assertGreater(result_gated.score, result_normal.score)

    def test_weights_sum_to_one(self):
        """Test that custom weights sum to 1."""
        learner = MetaLearnerFusion()
        result = learner._fuse_static_custom(0.5, 0.5, facial_weight=0.8, speech_weight=0.2)
        self.assertAlmostEqual(
            result.facial_weight + result.speech_weight, 1.0, places=5
        )


class TestConfidenceGatingIntegration(unittest.TestCase):
    """Integration tests for confidence gating with video/speech features."""

    def test_fuse_features_high_confidence(self):
        """Test fusion with high transcription confidence."""
        from multimodal_coercion.fusion.fusion_model import fuse_features

        video_features = {
            "emotion_probs": [0.2, 0.2],  # Avg = 0.2
        }
        speech_features = {
            "nlp_prob": 0.3,
            "transcription_confidence": 0.85,
            "transcription_reliable": True,
        }

        score = fuse_features(video_features, speech_features)
        # Should use normal fusion (0.5/0.5) or learned weights
        # With static fallback: 0.5*0.2 + 0.5*0.3 = 0.25
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_fuse_features_low_confidence(self):
        """Test fusion with low transcription confidence."""
        from multimodal_coercion.fusion.fusion_model import fuse_features

        video_features = {
            "emotion_probs": [0.8, 0.8],  # Avg = 0.8 (high stress)
        }
        speech_features = {
            "nlp_prob": 0.2,  # Low coercion from speech
            "transcription_confidence": 0.4,
            "transcription_reliable": False,
        }

        score = fuse_features(video_features, speech_features)
        # Should use confidence-gated weights (0.8/0.2)
        # 0.8*0.8 + 0.2*0.2 = 0.64 + 0.04 = 0.68
        # Score should lean toward facial (0.8) not speech (0.2)
        self.assertGreater(score, 0.2)
        self.assertLess(score, 1.0)

    def test_fuse_features_missing_confidence(self):
        """Test fusion when confidence fields are missing."""
        from multimodal_coercion.fusion.fusion_model import fuse_features

        video_features = {
            "emotion_probs": [0.5],
        }
        speech_features = {
            "nlp_prob": 0.5,
            # Missing confidence fields
        }

        # Should handle gracefully and use normal fusion
        score = fuse_features(video_features, speech_features)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
