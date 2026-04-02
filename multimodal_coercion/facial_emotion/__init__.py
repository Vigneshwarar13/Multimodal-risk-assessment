"""Facial emotion recognition module.

Provides TensorFlow-based emotion detection from facial images using
CNN model trained on FER2013 dataset. Includes Grad-CAM visualization
and temporal consistency analysis.

Modules:
    - tf_emotion_model: TensorFlow emotion classification model
    - face_detection: OpenCV-based face detection
    - features: Face feature extraction and preprocessing
    - preprocessing: Image normalization and augmentation
    - pipeline: High-level emotion inference pipeline
    - video_emotion_pipeline: Frame-level emotion tracking in videos
    - temporal_consistency: Temporal variance and drift analysis
"""
 
