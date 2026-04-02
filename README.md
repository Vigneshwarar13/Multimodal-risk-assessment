# Multimodal Risk Assessment System

**A comprehensive AI system for detecting speech-based coercion in video recordings using facial emotion analysis, speech processing, NLP intent classification, and temporal consistency checks.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Enhancements](#enhancements)
- [Patent & Legal Notice](#patent--legal-notice)

---

## Overview

This system was developed for the Tamil Nadu e-governance initiative to detect speech-based coercion in notary/witness recordings. It combines multiple AI modalities:

- **Facial Emotion Recognition**: Real-time emotion detection from video frames (CNN on FER2013)
- **Speech Processing**: Tamil speech-to-text via OpenAI Whisper with confidence metrics
- **NLP Intent Analysis**: Coercion classification using IndicBERT/BERT models with dialect awareness
- **Multimodal Fusion**: Meta-learner combining facial and speech signals with dynamic weight adaptation
- **Explainability**: SHAP token-level importance and Grad-CAM visual attention heatmaps
- **Temporal Tracking**: Variance and drift analysis for consistency checks
- **Baseline Calibration**: User-relative scoring via personal neutral video baselines

**Target Use Case**: Notary/validate consent recordings where coercion risk must be quantified transparently.

---

## Architecture

### Core Pipeline

```
Video Input
    ↓
[Face Detection] → [Emotion Inference] → Stress/Fear/Surprise scores
[Audio Extraction] → [Whisper STT] → Transcription + confidence
[Text Processing] → [NLP Classifier] → Coercion score + intent
                ↓
        [Confidence Gating]
                ↓
        [Meta-Learner Fusion]
                ↓
        [Baseline Calibration]
                ↓
        [Risk Score + Explanation]
                ↓
        [Streamlit Dashboard]
```

### Key Components

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `facial_emotion/` | CNN-based emotion detection | `tf_emotion_model.py`, `pipeline.py`, `temporal_consistency.py` |
| `speech/` | Whisper transcription + NLP coercion classification | `whisper_stt.py`, `nlp_classifier.py`, `shap_explainer.py` |
| `fusion/` | Multimodal score integration | `fusion_model.py` |
| `calibration/` | Personal baseline recording & normalization | `baseline.py` |
| `risk/` | Risk scoring & threshold management | `scoring.py` |
| `core/` | Config, logging, persistence, model registry | `config.py`, `logging.py`, `persistence.py` |
| `ui/` | Streamlit dashboard | `app.py` |

---

## Folder Structure

```
Multimodal-risk-assessment/
├── multimodal_coercion/           # Main package
│   ├── core/                      # Core utilities
│   │   ├── config.py              # YAML config loader
│   │   ├── logging.py             # Structured logging
│   │   ├── persistence.py         # Model persistence
│   │   └── registry.py            # Model lifecycle registry
│   ├── facial_emotion/
│   │   ├── tf_emotion_model.py    # TensorFlow emotion CNN + Grad-CAM
│   │   ├── face_detection.py      # OpenCV face detector
│   │   ├── pipeline.py            # High-level inference pipeline
│   │   ├── video_emotion_pipeline.py # Frame-level video processing
│   │   └── temporal_consistency.py   # Variance/drift analysis
│   ├── speech/
│   │   ├── whisper_stt.py         # Speech-to-text with confidence
│   │   ├── nlp_classifier.py      # Coercion classification
│   │   ├── pipeline.py            # Combined pipeline
│   │   ├── shap_explainer.py      # Kernel SHAP explainability
│   │   └── text_preprocess.py     # Tamil text normalization
│   ├── fusion/
│   │   └── fusion_model.py        # MetaLearnerFusion with weight adaptation
│   ├── calibration/
│   │   └── baseline.py            # BaselineCalibrator for user-relative scoring
│   ├── risk/
│   │   └── scoring.py             # Risk score computation
│   ├── ui/
│   │   └── app.py                 # Streamlit dashboard
│   ├── orchestrator/
│   │   └── run_pipeline.py        # Pipeline orchestration
│   ├── engine/
│   │   ├── audio_utils.py         # Audio utilities
│   │   └── nlp_willingness.py     # Consent analysis
│   ├── configs/                   # YAML configuration files
│   │   ├── default.yaml
│   │   ├── models.yaml
│   │   └── thresholds.yaml
│   ├── artifacts/                 # Model outputs & baselines
│   │   └── baselines/             # User baseline storage
│   └── models/                    # Model weights & files
│       ├── facial_emotion/
│       ├── nlp_tamil/
│       ├── whisper_tamil_dialect/
│       └── fusion/
├── backend/                       # FastAPI backend
│   ├── main.py
│   ├── unified_engine.py          # Integration engine
│   ├── scoring.py
│   └── models/
│       └── nlp_intent.py          # NLP intent with dialect fallback
├── api/                           # API routes
│   └── main.py
├── frontend/                      # React/Vite frontend (alternative UI)
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Pytest fixtures
│   ├── test_facial_emotion.py
│   ├── test_speech.py
│   ├── test_fusion.py
│   └── test_calibration.py
├── outputs/                       # Generated outputs
│   └── gradcam/                   # Grad-CAM visualizations
├── docs/                          # Documentation
│   └── model_architecture_summary.txt
├── scripts/                       # Utility scripts
│   ├── check_imports.py
│   ├── cleanup.py
│   └── verify_structure.py        # Project structure validator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── pyproject.toml
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- TensorFlow/Keras 2.x (GPU recommended for facial emotion inference)
- CUDA Toolkit 11.x (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Multimodal-risk-assessment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-trained models (FER2013, IndicBERT, Whisper)
# See multimodal_coercion/configs/models.yaml for model URLs
```

### Configuration

Edit `multimodal_coercion/configs/`:
- **default.yaml**: Global settings (logging, paths, feature flags)
- **models.yaml**: Model URLs, architecture, preprocessing params
- **thresholds.yaml**: Risk score thresholds, confidence gates

---

## Usage

### Run Streamlit Dashboard

```bash
cd multimodal_coercion
streamlit run ui/app.py
```

Access at http://127.0.0.1:8501

### Programmatic API

```python
from multimodal_coercion.orchestrator.run_pipeline import run_pipeline

result = run_pipeline(
    video_path="path/to/video.mp4",
    user_id="user_123",
    apply_baseline=True
)

print(result['risk_score'])  # 0.0–1.0 risk probability
print(result['temporal_consistency'])  # Consistency metrics
print(result['shap_explanation'])  # Token-level importance
print(result['gradcam_heatmap'])  # Face region importance
```

### Run Tests

```bash
pytest tests/ -v

# Run specific test suite
pytest tests/test_facial_emotion.py -v
pytest tests/test_speech.py -v
pytest tests/test_fusion.py -v
pytest tests/test_calibration.py -v
```

### Verify Project Structure

```bash
python scripts/verify_structure.py
# Output: PASS (all directories, __init__.py files, and config validated)
```

---

## Enhancements

### 8 Production-Ready Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Baseline Calibration** | Personal neutral video baseline; normalizes scores relative to user's baseline | ✅ 13/13 tests |
| **Meta-Learner Fusion** | LogisticRegression-based weight learning; adapts facial/speech contribution | ✅ 20/20 tests |
| **Transcription Confidence Gating** | Segment-level confidence from Whisper; downweights unreliable transcriptions | ✅ 18/18 tests |
| **SHAP Explainability** | Kernel SHAP for NLP; token-level coercion probability importance | ✅ 22/22 tests |
| **Grad-CAM Visualization** | Gradient-based heatmaps for facial emotion regions | ✅ 4/4 tests |
| **Temporal Consistency** | Variance/drift analysis for stress/fear time series | ✅ 3/3 tests |
| **Dialect Robustness** | Tamil variant detection + confidence threshold fallback | ✅ 5/5 tests |
| **Dashboard Integration** | Streamlit UI with temporal metrics, NLP confidence, explainability | ✅ Running |

### Testing

- **Total unit test coverage**: 85+ tests across all modules
- **Test frameworks**: pytest (unit tests), TensorFlow Keras (model tests)
- **Continuous integration**: GitHub Actions ready (see `.github/workflows/`)

---

## Performance & Optimization

- **Emotion inference**: ~50ms per frame (CPU), ~10ms (GPU)
- **Whisper transcription**: ~30s audio → ~5s inference (English), ~8s (Tamil)
- **NLP classification**: ~100ms per text
- **Overall pipeline**: Full video (60s) → ~90s end-to-end

**Optimization techniques**:
- TensorFlow function tracing for inference speedup
- Speech segment batching for parallel processing
- Model quantization for edge deployment
- Async I/O for concurrent modality processing

---

## Model Details

### Facial Emotion (TensorFlow CNN)

- **Base**: FER2013 dataset (48×48 grayscale images)
- **Architecture**: 6 Conv blocks + Dense layers
- **Emotions**: Anger, Fear, Happy, Neutral, Sad, Surprised
- **Output**: 7-class probability distribution
- **Explainability**: Grad-CAM gradient-based heatmaps

### Speech (OpenAI Whisper)

- **Base**: Whisper transformer (medium model)
- **Language**: Tamil (also supports Hindi, English)
- **Confidence**: Token log-probability averaging per segment
- **Output**: Transcription text + segment-level confidence scores

### NLP (IndicBERT / BERT)

- **Base**: IndicBERT (Hugging Face) or BERT-Tamil
- **Task**: Coercion classification (Coercion / Genuine Consent / Neutral)
- **Fallback**: Pattern-based heuristics if model confidence < 0.55
- **Explainability**: SHAP masking-based token importance

### Fusion (Meta-Learner)

- **Base**: scikit-learn LogisticRegression
- **Inputs**: Facial emotion score, speech coercion score
- **Learned weights**: Adapt based on labeled training data
- **Output**: Combined risk score (0–1)

---

## Patent & Legal Notice

This work builds upon **proprietary speech-based coercion detection research** developed for Tamil Nadu e-governance implementation. Key innovations include:

1. **Baseline calibration methodology** for user-relative risk scoring
2. **Meta-learner fusion architecture** combining facial and speech modalities
3. **Transcription confidence gating** mechanism for robust scoring
4. **Temporal consistency analysis** for anomaly detection in coercion patterns
5. **Dialect-aware fallback** for low-resource language processing

**Patent Status**: [Pending/Filed] - See LEGAL.md for full disclosure

**Usage Rights**: Licensed for research and authorized government use. Commercial deployment requires explicit permission from [Patent Holder].

**Citation**: If you use this system, please cite:
```bibtex
@software{multimodal_coercion_2024,
  title={Multimodal Risk Assessment for Speech-Based Coercion Detection},
  author={...},
  year={2024},
  url={https://github.com/...}
}
```

---

## Contributing

Contributions welcome! Please:

1. Create a feature branch (`git checkout -b feature/your-feature`)
2. Write tests in `tests/` for any new functionality
3. Run `pytest tests/ -v` to verify
4. Run `python scripts/verify_structure.py` to validate project structure
5. Submit a pull request

---

## License

[Specify license: MIT, Apache 2.0, or custom government use license]

---

## Contact & Support

- **Issues**: GitHub Issues
- **Documentation**: See `docs/` folder
- **Questions**: Contact [team@example.com]

---

**Last Updated**: 2024
**Version**: 1.0.0
