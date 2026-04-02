"""Fusion module for multimodal integration.

Combines facial emotion and speech scores using meta-learner approach.
Includes dynamic weight adaptation via LogisticRegression and confidence
gating based on transcription reliability.

Modules:
    - fusion_model: MetaLearnerFusion class with weight learning
""" 
