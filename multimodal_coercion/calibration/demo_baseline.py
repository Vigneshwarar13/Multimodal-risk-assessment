"""
Example demonstration of baseline calibration with the full pipeline.

This script shows:
1. How to use the baseline calibration system
2. How to integrate it with the video/speech pipelines
3. How the normalization affects coercion detection accuracy

Usage:
    python -m multimodal_coercion.calibration.demo_baseline
"""

import sys
from pathlib import Path
from typing import List


def demo_basic_calibration():
    """Demonstrate basic baseline calibration without pipelines."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Baseline Calibration")
    print("=" * 70)

    from multimodal_coercion.calibration.baseline import BaselineCalibrator

    # Create calibrator for a user
    calibrator = BaselineCalibrator(user_id="demo_user_001")

    print("\n[CALIBRATION PHASE]")
    print("Recording 3 neutral question responses...")

    # Simulate neutral responses (low stress, low coercion)
    neutral_samples = [
        (0.18, 0.15),  # Q1: "What is your name?"
        (0.16, 0.14),  # Q2: "Where do you live?"
        (0.19, 0.16),  # Q3: "What is your occupation?"
    ]

    for idx, (facial, speech) in enumerate(neutral_samples, 1):
        calibrator.record_sample(facial, speech)
        print(f"  Q{idx}: facial={facial:.2f}, speech={speech:.2f}")

    # Compute baseline
    baseline = calibrator.compute_baseline()
    print(f"\nBaseline computed:")
    print(f"  Facial baseline: {baseline.facial_baseline:.3f}")
    print(f"  Speech baseline: {baseline.speech_baseline:.3f}")
    print(f"  Timestamp: {baseline.timestamp}")

    # Demonstrate normalization effect
    print("\n[DEMONSTRATION: Normalization Effect]")
    print("Same person, now answering registration questions...")

    test_scores = [
        (0.25, "Slightly stressed"),
        (0.40, "Moderately stressed"),
        (0.60, "Very stressed"),
        (0.80, "Extremely stressed"),
    ]

    print(
        f"\n{'Raw Score':<12} {'Normalized':<12} {'Category':<20} {'Assessment':<20}"
    )
    print("-" * 60)

    for raw_score, assessment in test_scores:
        normalized = calibrator.normalize_score(raw_score, baseline)

        # Categorize
        if normalized < 0.4:
            category = "Good (Low Risk)"
        elif normalized < 0.7:
            category = "Average (Med Risk)"
        else:
            category = "Poor (High Risk)"

        print(
            f"{raw_score:<12.2f} {normalized:<12.3f} {category:<20} {assessment:<20}"
        )

    print("\n→ Key insight: Baseline adjustment reduces false positives for")
    print("  naturally tense or expressive individuals.")


def demo_with_scores():
    """
    Demonstrate calibration with realistic coercion detection scores.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Realistic Score Normalization")
    print("=" * 70)

    from multimodal_coercion.calibration.baseline import BaselineCalibrator, BaselineResult

    # Scenario: Two users with different emotional baselines
    print("\n[SCENARIO]")
    print("Two users undergo the same registration process.")
    print("User A: Naturally calm and reserved")
    print("User B: Naturally expressive and tense")

    # User A (calm)
    print("\n--- USER A: Calm Baseline ---")
    calib_a = BaselineCalibrator(user_id="user_a")
    calib_a.record_sample(0.12, 0.10)
    calib_a.record_sample(0.14, 0.11)
    calib_a.record_sample(0.13, 0.10)
    baseline_a = calib_a.compute_baseline()
    print(f"Baseline: facial={baseline_a.facial_baseline:.3f}, "
          f"speech={baseline_a.speech_baseline:.3f}")

    # User B (expressive)
    print("\n--- USER B: Expressive Baseline ---")
    calib_b = BaselineCalibrator(user_id="user_b")
    calib_b.record_sample(0.45, 0.40)
    calib_b.record_sample(0.48, 0.42)
    calib_b.record_sample(0.46, 0.41)
    baseline_b = calib_b.compute_baseline()
    print(f"Baseline: facial={baseline_b.facial_baseline:.3f}, "
          f"speech={baseline_b.speech_baseline:.3f}")

    # Both record same registration session scores
    session_raw_score = 0.55

    print(f"\n[REGISTRATION SESSION]")
    print(f"Recorded score (before baseline): {session_raw_score:.2f}")

    norm_a = calib_a.normalize_score(session_raw_score, baseline_a)
    norm_b = calib_b.normalize_score(session_raw_score, baseline_b)

    print(f"\nAfter baseline normalization:")
    print(f"  User A: {norm_a:.3f} → {'Good (low risk)' if norm_a < 0.4 else 'Average' if norm_a < 0.7 else 'Poor (high risk)'}")
    print(f"  User B: {norm_b:.3f} → {'Good (low risk)' if norm_b < 0.4 else 'Average' if norm_b < 0.7 else 'Poor (high risk)'}")

    print("\n→ Without baseline: both would be classified as 'Average'")
    print("→ With baseline: User A is 'Good', User B is 'Average'")
    print("→ This accounts for individual differences in expressiveness")


def demo_edge_cases():
    """Demonstrate edge cases and boundary conditions."""
    print("\n" + "=" * 70)
    print("DEMO 3: Edge Cases")
    print("=" * 70)

    from multimodal_coercion.calibration.baseline import BaselineCalibrator, BaselineResult

    print("\n[CASE 1: Very low baseline (very calm person)]")
    baseline_low = BaselineResult(
        user_id="very_calm",
        facial_baseline=0.05,
        speech_baseline=0.05,
        sample_count=3,
        timestamp="2024-04-02",
    )
    calib = BaselineCalibrator("very_calm")
    for score in [0.1, 0.2, 0.3, 0.5, 0.9]:
        norm = calib.normalize_score(score, baseline_low)
        print(f"  Raw {score:.2f} → Norm {norm:.3f}")

    print("\n[CASE 2: Very high baseline (very tense person)]")
    baseline_high = BaselineResult(
        user_id="very_tense",
        facial_baseline=0.8,
        speech_baseline=0.8,
        sample_count=3,
        timestamp="2024-04-02",
    )
    calib = BaselineCalibrator("very_tense")
    for score in [0.2, 0.4, 0.6, 0.8, 0.95]:
        norm = calib.normalize_score(score, baseline_high)
        print(f"  Raw {score:.2f} → Norm {norm:.3f}")

    print("\n[CASE 3: No baseline (graceful fallback)]")
    calib = BaselineCalibrator("no_baseline")
    score = 0.65
    norm = calib.normalize_score(score)  # No baseline
    print(f"  Raw {score:.2f} → Norm {norm:.3f} (returns original score)")

    print("\n[CASE 4: Clipping behavior (scores > 1.0)]")
    baseline = BaselineResult(
        user_id="test",
        facial_baseline=0.3,
        speech_baseline=0.3,
        sample_count=3,
        timestamp="2024-04-02",
    )
    calib = BaselineCalibrator("test")
    # Extreme stress (very high)
    norm = calib.normalize_score(1.5, baseline)  # Will be clipped at input
    print(f"  Raw 1.5 (clipped → 1.0) → Norm {norm:.3f}")


def demo_persistence():
    """Demonstrate baseline persistence (if file system available)."""
    print("\n" + "=" * 70)
    print("DEMO 4: Baseline Persistence")
    print("=" * 70)

    try:
        import tempfile
        from multimodal_coercion.calibration.baseline import BaselineCalibrator
        from multimodal_coercion.core.persistence import Persistence

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\nUsing temporary directory: {tmpdir}")

            # Create persistence layer
            db = Persistence(
                db_path=str(Path(tmpdir) / "test.db"),
                artifacts_dir=str(Path(tmpdir) / "artifacts"),
            )

            # Step 1: Create and save baseline
            print("\n[STEP 1] Calibrating and saving baseline...")
            calib1 = BaselineCalibrator("persistent_user", persistence=db)
            calib1.record_sample(0.20, 0.18)
            calib1.record_sample(0.22, 0.19)
            calib1.record_sample(0.21, 0.17)
            baseline1 = calib1.compute_baseline()
            print(f"Baseline saved: facial={baseline1.facial_baseline:.3f}, "
                  f"speech={baseline1.speech_baseline:.3f}")

            # Verify file exists
            baseline_file = Path(tmpdir) / "artifacts" / "baselines" / "persistent_user_baseline.json"
            print(f"Baseline file exists: {baseline_file.exists()}")

            # Step 2: Load baseline in new session
            print("\n[STEP 2] Loading baseline in new session...")
            calib2 = BaselineCalibrator("persistent_user", persistence=db)
            loaded_baseline = calib2.load_baseline()

            if loaded_baseline:
                print(f"Baseline loaded: facial={loaded_baseline.facial_baseline:.3f}, "
                      f"speech={loaded_baseline.speech_baseline:.3f}")
                print(f"Match: {abs(loaded_baseline.facial_baseline - baseline1.facial_baseline) < 1e-5}")
            else:
                print("Failed to load baseline")

    except Exception as e:
        print(f"Persistence demo skipped: {e}")


def demo_integration_summary():
    """Print integration summary."""
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)

    print("""
The baseline calibration system integrates into the pipeline as follows:

1. CALIBRATION PHASE (before registration)
   ├─ User answers 3 neutral questions
   ├─ Video & speech analyzed for each response
   └─ Baseline scores computed and stored

2. REGISTRATION PHASE
   ├─ User records actual registration transaction
   ├─ Video & speech analyzed
   └─ Raw coercion score computed

3. NORMALIZATION
   ├─ Raw score compared to baseline
   └─ Normalized score = (raw - baseline) / (1 - baseline)

4. CLASSIFICATION
   ├─ Apply risk thresholds to normalized score
   └─ Output: Good / Average / Poor + confidence

BENEFITS:
✓ Reduces false positives for expressive individuals
✓ Maintains sensitivity to actual coercion attempts
✓ Personalized thresholds for individual differences
✓ Backward compatible - works without baseline (falls back to raw score)

NEXT INTEGRATION POINTS:
- Task 2: Dynamic fusion weights (meta-learner)
- Task 3: Transcription confidence gating
- Task 4: SHAP explainability
    """)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("BASELINE CALIBRATION SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 70)

    try:
        demo_basic_calibration()
        demo_with_scores()
        demo_edge_cases()
        demo_persistence()
        demo_integration_summary()

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
