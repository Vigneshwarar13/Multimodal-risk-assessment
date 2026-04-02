"""Multimodal Coercion Detection System

A comprehensive system for detecting speech-based coercion in video recordings
using facial emotion analysis, speech transcription, NLP intent classification,
and temporal consistency checks.

Modules:
    - core: Configuration, logging, and persistence utilities
    - engine: Main orchestration and inference pipeline
    - facial_emotion: TensorFlow-based emotion recognition from faces
    - speech: Whisper-based Tamil speech-to-text with confidence metrics
    - fusion: Meta-learner fusion model combining modalities
    - calibration: Personal baseline calibration for user-relative scoring
    - risk: Risk scoring and threshold management
    - orchestrator: High-level pipeline orchestration
    - ui: Streamlit dashboard interface
"""

__version__ = "1.0.0"
__author__ = "Multimodal Risk Assessment Team"
