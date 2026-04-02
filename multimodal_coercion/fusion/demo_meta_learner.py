"""
Interactive demonstration of dynamic fusion weights with meta-learner.

Shows:
1. Training a meta-learner on labeled data
2. Using static fallback weights
3. Using learned weights for fusion
4. Comparing results
"""

import sys
import numpy as np
from pathlib import Path
import tempfile

from multimodal_coercion.fusion.fusion_model import (
    train_meta_learner,
    load_and_fuse,
    MetaLearnerFusion,
    fuse_scores,
    fuse_scores_with_confidence,
)


def demo_synthetic_training_data():
    """Generate realistic synthetic training data."""
    print("\n" + "=" * 70)
    print("DEMO 1: Synthetic Training Data")
    print("=" * 70)

    # Scenario: 50 coercion cases, 50 genuine consent cases
    np.random.seed(42)

    # Genuine consent: low facial stress, low speech coercion
    genuine_facial = np.random.normal(0.15, 0.08, 50)
    genuine_speech = np.random.normal(0.12, 0.07, 50)
    genuine_labels = np.zeros(50, dtype=int)

    # Coercion: high facial stress, high speech coercion
    coerced_facial = np.random.normal(0.75, 0.1, 50)
    coerced_speech = np.random.normal(0.72, 0.09, 50)
    coerced_labels = np.ones(50, dtype=int)

    # Combine
    facial_scores = np.concatenate([genuine_facial, coerced_facial])
    speech_scores = np.concatenate([genuine_speech, coerced_speech])
    labels = np.concatenate([genuine_labels, coerced_labels])

    print(f"\nGenerated {len(facial_scores)} training samples:")
    print(f"  Genuine consent: {np.sum(labels == 0)} samples")
    print(f"  Coercion: {np.sum(labels == 1)} samples")

    print(f"\nGemuine consent scores:")
    print(f"  Facial: avg={genuine_facial.mean():.3f}, std={genuine_facial.std():.3f}")
    print(f"  Speech: avg={genuine_speech.mean():.3f}, std={genuine_speech.std():.3f}")

    print(f"\nCoercion scores:")
    print(f"  Facial: avg={coerced_facial.mean():.3f}, std={coerced_facial.std():.3f}")
    print(f"  Speech: avg={coerced_speech.mean():.3f}, std={coerced_speech.std():.3f}")

    return facial_scores, speech_scores, labels


def demo_static_vs_learned_fusion():
    """Compare static weights vs learned weights."""
    print("\n" + "=" * 70)
    print("DEMO 2: Static vs Learned Fusion Weights")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate training data
        facial_scores, speech_scores, labels = demo_synthetic_training_data()

        # Train meta-learner
        print("\n[TRAINING META-LEARNER]")
        summary = train_meta_learner(
            facial_scores, speech_scores, labels, model_dir=tmpdir
        )

        print(f"Training completed:")
        print(f"  Facial weight: {summary['facial_weight']:.3f}")
        print(f"  Speech weight: {summary['speech_weight']:.3f}")
        print(f"  Model saved to: {summary['model_path']}")

        # Create test cases
        test_cases = [
            (0.1, 0.1, "Genuine - low stress"),
            (0.2, 0.15, "Genuine - moderate stress"),
            (0.4, 0.35, "Ambiguous - mixed signals"),
            (0.7, 0.65, "Coerced - high stress"),
            (0.85, 0.80, "Coerced - very high stress"),
        ]

        print("\n[COMPARISON: Static vs Learned Weights]")
        print(
            f"\n{'Test Case':<30} {'Facial':<8} {'Speech':<8} "
            f"{'Static':<10} {'Learned':<10} {'Confidence':<12}"
        )
        print("-" * 80)

        # Test each case with both methods
        learner = MetaLearnerFusion(model_dir=tmpdir)
        learner.load_model()

        for facial, speech, desc in test_cases:
            # Static fusion
            result_static = learner._fuse_static(facial, speech)

            # Learned fusion
            result_learned = learner.fuse(facial, speech)

            print(
                f"{desc:<30} {facial:<8.2f} {speech:<8.2f} "
                f"{result_static.score:<10.3f} {result_learned.score:<10.3f} "
                f"{result_learned.confidence:<12.1%}"
            )

        print("\n→ Key insight: Learned weights adapt to data patterns")
        print("→ Static fallback available if model not trained")


def demo_learned_weight_dynamics():
    """Show how learned weights vary based on input patterns."""
    print("\n" + "=" * 70)
    print("DEMO 3: Learned Weight Dynamics")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Scenario: Facial more predictive than speech
        print("\n[Scenario: Facial stress is more predictive]")

        # Genuine: both low
        genuine_facial = np.array([0.1, 0.12, 0.15, 0.18, 0.14])
        genuine_speech = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Noise
        genuine_labels = np.zeros(5, dtype=int)

        # Coerced: facial high, speech still mixed
        coerced_facial = np.array([0.8, 0.82, 0.85, 0.78, 0.84])
        coerced_speech = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Noise
        coerced_labels = np.ones(5, dtype=int)

        facial_scores = np.concatenate([genuine_facial, coerced_facial])
        speech_scores = np.concatenate([genuine_speech, coerced_speech])
        labels = np.concatenate([genuine_labels, coerced_labels])

        summary = train_meta_learner(
            facial_scores, speech_scores, labels, model_dir=tmpdir
        )

        print(f"\nTrained weights:")
        print(f"  Facial weight: {summary['facial_weight']:.3f} (higher = more predictive)")
        print(f"  Speech weight: {summary['speech_weight']:.3f}")
        print(f"→ Model learned facial is more predictive!\n")

        # Now reverse: speech more predictive
        print("\n[Scenario: Speech coercion is more predictive]")

        # Genuine: both low
        genuine_facial = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Noise
        genuine_speech = np.array([0.1, 0.12, 0.15, 0.18, 0.14])
        genuine_labels = np.zeros(5, dtype=int)

        # Coerced: speech high, facial still mixed
        coerced_facial = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Noise
        coerced_speech = np.array([0.8, 0.82, 0.85, 0.78, 0.84])
        coerced_labels = np.ones(5, dtype=int)

        facial_scores = np.concatenate([genuine_facial, coerced_facial])
        speech_scores = np.concatenate([genuine_speech, coerced_speech])
        labels = np.concatenate([genuine_labels, coerced_labels])

        summary = train_meta_learner(
            facial_scores, speech_scores, labels, model_dir=tmpdir + "_2"
        )

        print(f"\nTrained weights:")
        print(f"  Facial weight: {summary['facial_weight']:.3f}")
        print(f"  Speech weight: {summary['speech_weight']:.3f} (higher = more predictive)")
        print(f"→ Model learned speech is more predictive!")


def demo_confidence_estimation():
    """Show confidence estimation in predictions."""
    print("\n" + "=" * 70)
    print("DEMO 4: Confidence Estimation")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate training data
        facial_scores, speech_scores, labels = demo_synthetic_training_data()
        train_meta_learner(facial_scores, speech_scores, labels, model_dir=tmpdir)

        # Test confidence at various agreement levels
        print("\n[Confidence with varying input agreement]")
        print("\nHigh agreement (both modalities agree):")

        learner = MetaLearnerFusion(model_dir=tmpdir)
        learner.load_model()

        result = learner.fuse(0.1, 0.1)
        print(f"  Scores: facial=0.10, speech=0.10")
        print(f"  Prediction: {result.score:.3f}, Confidence: {result.confidence:.1%}")

        result = learner.fuse(0.9, 0.9)
        print(f"  Scores: facial=0.90, speech=0.90")
        print(f"  Prediction: {result.score:.3f}, Confidence: {result.confidence:.1%}")

        print("\nLow agreement (contradictory signals):")

        result = learner.fuse(0.1, 0.9)
        print(f"  Scores: facial=0.10, speech=0.90")
        print(f"  Prediction: {result.score:.3f}, Confidence: {result.confidence:.1%}")

        result = learner.fuse(0.9, 0.1)
        print(f"  Scores: facial=0.90, speech=0.10")
        print(f"  Prediction: {result.score:.3f}, Confidence: {result.confidence:.1%}")

        print("\n→ Lower confidence when modalities disagree")
        print("→ Higher confidence when modalities agree")


def demo_integration_with_pipeline():
    """Show integration with existing pipeline."""
    print("\n" + "=" * 70)
    print("DEMO 5: Integration with Existing Pipeline")
    print("=" * 70)

    print("""
Before (static weights):
    score = 0.5 * facial_score + 0.5 * speech_score

After (with meta-learner):
    result = load_and_fuse(facial_score, speech_score)
    score = result.score  # Learned weights applied
    confidence = result.confidence
    weights = (result.facial_weight, result.speech_weight)

Integration in orchestrator:
    1. If trained model exists, use learned weights
    2. Otherwise, fallback to static 0.5/0.5 weights
    3. Return both score and confidence in result

Benefits:
    ✓ Automatic weight optimization from data
    ✓ Confidence estimates for reliability assessment
    ✓ Backward compatible (fallback to static)
    ✓ No changes to video/speech pipelines needed
    ✓ Weights can be retrained as new labeled data arrives
    """)

    print("\nExample usage in pipeline:")
    print("""
from multimodal_coercion.fusion.fusion_model import load_and_fuse

# In run_full_pipeline
facial_score = video_feats['emotion_score']
speech_score = speech_feats['nlp_prob']

# Original: score = 0.5 * facial + 0.5 * speech
# New:
result = load_and_fuse(facial_score, speech_score)
score = result.score
confidence = result.confidence

# Result includes:
# - score: final coercion probability
# - confidence: model confidence in prediction
# - facial_weight: contribution of facial modality
# - speech_weight: contribution of speech modality
# - model_used: "learned" or "static"
    """)


def demo_training_workflow():
    """Show typical training workflow."""
    print("\n" + "=" * 70)
    print("DEMO 6: Training Workflow")
    print("=" * 70)

    print("""
Step 1: Collect labeled data
    - Video/speech analysis on 100+ registration sessions
    - Ground truth labels: genuine consent or coercion
    - Extract facial_scores and speech_scores for each

Step 2: Train meta-learner
    from multimodal_coercion.fusion.fusion_model import train_meta_learner
    
    summary = train_meta_learner(
        facial_scores=np.array([...]),      # (n_samples,)
        speech_scores=np.array([...]),      # (n_samples,)
        labels=np.array([0, 1, 0, ...]),    # 0=genuine, 1=coercion
        model_dir="models/fusion/"
    )
    
    print(f"Facial weight: {summary['facial_weight']:.3f}")
    print(f"Speech weight: {summary['speech_weight']:.3f}")

Step 3: Use in production
    - Model automatically loaded when pipeline starts
    - If model not found, graceful fallback to 0.5/0.5 weights
    - Weights continuously applied to all new registrations

Step 4: Monitor and retrain
    - Collect new labeled sessions
    - Periodically retrain with updated data
    - Model weights adapt to actual coercion patterns in region

KEY FILES:
    - models/fusion/meta_learner.pkl → Trained model (joblib format)
    - multimodal_coercion/fusion/fusion_model.py → Main implementation
    - Test: python -m unittest multimodal_coercion.fusion.test_fusion_meta_learner
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("DYNAMIC FUSION WEIGHTS - META-LEARNER DEMONSTRATION")
    print("=" * 70)

    try:
        demo_static_vs_learned_fusion()
        demo_learned_weight_dynamics()
        demo_confidence_estimation()
        demo_integration_with_pipeline()
        demo_training_workflow()

        print("\n" + "=" * 70)
        print("✓ All demonstrations completed successfully")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
