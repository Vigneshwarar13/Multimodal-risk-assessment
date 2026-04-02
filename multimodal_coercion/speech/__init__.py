"""Speech processing module.

Provides Tamil speech-to-text transcription via OpenAI Whisper,
Intent classification using IndicBERT/BERT models, and SHAP-based
token-level explainability.

Modules:
    - whisper_stt: Speech-to-text conversion with segment confidence
    - nlp_classifier: Tamil coercion intent classification
    - text_preprocess: Tamil text normalization and preprocessing
    - pipeline: Combined transcription and intent pipeline
    - shap_explainer: Kernel SHAP for token-level importance
""" 
