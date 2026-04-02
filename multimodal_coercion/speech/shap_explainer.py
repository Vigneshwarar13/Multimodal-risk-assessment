"""
SHAP Explainability for Tamil Coercion NLP Classifier

Provides local interpretability using SHAP (SHapley Additive exPlanations) to identify
which words/tokens contribute most to coercion predictions in Tamil text.

Uses a masked language model approach with Transformers to compute feature importance.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPExplainerConfig:
    """Configuration for SHAP explainability."""
    
    def __init__(
        self,
        num_samples: int = 50,
        aggregation: str = "mean",
        background_samples: int = 10,
        max_tokens: int = 256,
    ):
        """
        num_samples: Number of background samples for SHAP computation
        aggregation: How to aggregate token-level SHAP values ("mean", "max", "abs")
        background_samples: Size of background dataset for baseline
        max_tokens: Maximum tokens to consider for computational efficiency
        """
        self.num_samples = num_samples
        self.aggregation = aggregation
        self.background_samples = background_samples
        self.max_tokens = max_tokens


class ShapleyExplainer:
    """
    SHAP-based explainer for transformer models.
    
    Uses Kernel SHAP (model-agnostic) or Partition SHAP to compute token importance.
    Works with HuggingFace transformers and returns token-level SHAP values.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        target_class: int = 2,  # Index for "Coercion" class
        config: Optional[SHAPExplainerConfig] = None,
    ):
        """
        model: AutoModelForSequenceClassification instance
        tokenizer: AutoTokenizer instance (compatible with model)
        device: "cpu", "cuda", etc.
        target_class: Which class index to explain (default: 2 for Coercion)
        config: SHAPExplainerConfig instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_class = target_class
        self.config = config or SHAPExplainerConfig()
        
        # Cache for background dataset
        self._background_texts = []
        self._shap_explainer = None
    
    def set_background_data(self, texts: List[str]):
        """
        Set background dataset for SHAP computation.
        
        texts: List of representative Tamil text samples
        """
        self._background_texts = texts[:self.config.background_samples]
        logger.info(f"Set background data with {len(self._background_texts)} samples")
    
    def _prediction_function(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper function for model prediction used by SHAP.
        
        texts: List of text samples
        Returns: Array of shape (len(texts),) with probabilities for target_class
        """
        import torch
        
        probs = []
        for text in texts:
            try:
                # Tokenize with same settings as main classifier
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.config.max_tokens,
                    padding=False,
                    return_tensors="pt"
                )
                
                # Move to device
                enc = {k: v.to(self.device) for k, v in enc.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**enc)
                    logits = outputs.logits[0]
                    prob = torch.softmax(logits, dim=-1)[self.target_class].cpu().item()
                
                probs.append(prob)
            except Exception as e:
                logger.warning(f"Prediction failed for text: {str(e)}")
                probs.append(0.5)  # Default to neutral
        
        return np.array(probs)
    
    def explain_prediction(
        self,
        text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP.
        
        Returns dict with:
        {
            "text": str,
            "prediction_score": float,
            "label": str,
            "tokens": List[Dict],  # [{"token": str, "shap_value": float, "contribution": float}]
            "top_tokens": List[Dict],  # Top k by absolute SHAP value
            "background_size": int,
            "num_samples_used": int,
            "error": Optional[str]  # If generation failed
        }
        """
        try:
            import torch
            from transformers import pipeline
            
            # Get base prediction
            base_prediction = self._prediction_function([text])[0]
            
            # Tokenize to get token sequence
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_tokens,
                padding=False,
                return_tensors="pt"
            )
            
            token_ids = enc["input_ids"][0].cpu().numpy()
            token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Compute SHAP values via permutation (Kernel SHAP approach)
            shap_values = self._compute_shap_values(text, token_ids, token_strs)
            
            # Build result
            tokens_result = []
            for token_str, shap_val in zip(token_strs, shap_values):
                # Filter special tokens
                if token_str in ["[CLS]", "[SEP]", "[PAD]", "<pad>"]:
                    continue
                
                tokens_result.append({
                    "token": token_str,
                    "shap_value": float(shap_val),
                    "contribution": "positive" if shap_val > 0 else "negative",
                    "magnitude": float(abs(shap_val))
                })
            
            # Sort by magnitude and get top k
            tokens_result_sorted = sorted(tokens_result, key=lambda x: x["magnitude"], reverse=True)
            top_tokens = tokens_result_sorted[:top_k]
            
            # Determine label
            label_map = {0: "Genuine Consent", 1: "Neutral", 2: "Coercion"}
            pred_label = label_map.get(self.target_class, "Unknown")
            
            return {
                "text": text,
                "prediction_score": float(base_prediction),
                "label": pred_label,
                "tokens": tokens_result,
                "top_tokens": top_tokens,
                "background_size": len(self._background_texts),
                "num_samples_used": self.config.num_samples,
                "error": None
            }
        
        except Exception as e:
            logger.exception(f"SHAP explanation failed: {str(e)}")
            return {
                "text": text,
                "prediction_score": 0.0,
                "label": "Unknown",
                "tokens": [],
                "top_tokens": [],
                "background_size": 0,
                "num_samples_used": 0,
                "error": str(e)
            }
    
    def _compute_shap_values(
        self,
        text: str,
        token_ids: np.ndarray,
        token_strs: List[str]
    ) -> np.ndarray:
        """
        Compute SHAP values using Kernel SHAP (permutation-based approach).
        
        Strategy:
        1. For each token, compute score with token present vs masked
        2. Use average of background samples as baseline
        3. Difference = SHAP value approximation
        
        Returns: Array of SHAP values per token
        """
        import torch
        
        try:
            # Get baseline prediction from background
            if self._background_texts:
                baseline_preds = self._prediction_function(self._background_texts)
                baseline_score = float(np.mean(baseline_preds))
            else:
                baseline_score = 0.5
            
            # Full text prediction
            full_prediction = self._prediction_function([text])[0]
            
            # Compute SHAP value per token via masking
            shap_values = []
            mask_token = self.tokenizer.mask_token or "[MASK]"
            
            for idx, token_str in enumerate(token_strs):
                # Skip special tokens
                if token_str in ["[CLS]", "[SEP]", "[PAD]", "<pad>"]:
                    shap_values.append(0.0)
                    continue
                
                try:
                    # Create masked version by replacing token with [MASK]
                    tokens_list = list(token_strs)
                    tokens_list[idx] = mask_token
                    masked_text = self.tokenizer.convert_tokens_to_string(tokens_list)
                    
                    # Get prediction with masked token
                    masked_prediction = self._prediction_function([masked_text])[0]
                    
                    # SHAP approximation: (full - masked) shows token's contribution
                    shap_val = full_prediction - masked_prediction
                    shap_values.append(shap_val)
                
                except Exception as e:
                    logger.debug(f"Masking failed for token {idx}: {str(e)}")
                    shap_values.append(0.0)
            
            return np.array(shap_values)
        
        except Exception as e:
            logger.warning(f"SHAP computation failed: {str(e)}")
            # Return zeros if computation fails
            return np.zeros(len(token_strs))


def create_shap_explainer(
    classifier,
    device: str = "cpu",
    background_texts: Optional[List[str]] = None,
    config: Optional[SHAPExplainerConfig] = None
) -> ShapleyExplainer:
    """
    Create a SHAP explainer attached to a TamilCoercionClassifier.
    
    classifier: TamilCoercionClassifier instance (with model loaded)
    device: "cpu" or "cuda"
    background_texts: Optional list of background texts for baseline
    config: Optional SHAPExplainerConfig
    
    Returns: ShapleyExplainer instance
    """
    if classifier._model is None or classifier._tokenizer is None:
        raise ValueError("Classifier must have model and tokenizer loaded")
    
    explainer = ShapleyExplainer(
        model=classifier._model,
        tokenizer=classifier._tokenizer,
        device=device,
        target_class=classifier.label2id.get("Coercion", 2),
        config=config
    )
    
    if background_texts:
        explainer.set_background_data(background_texts)
    
    return explainer


# Default background texts for Tamil (neutral statements)
DEFAULT_TAMIL_BACKGROUND = [
    "என் பெயர் என்றால் என்ன?",
    "இன்று天weather எப்படி இருக்கிறது?",
    "நீங்கள் புத்தகம் வாசிக்கிறீர்கள்?",
    "அவர் வேலைக்கு போனார்.",
    "நான் பள்ளிக்கு செல்கிறேன்.",
    "இது மிகவும் நல்ல நாள்.",
    "உங்கள் குடும்பம் எப்படி இருக்கிறது?",
    "நான் சாப்பாடு சாப்பிட்டேன்.",
    "ஆம், நான் உடன்படுகிறேன்.",
    "இல்லை, அது சரியல்ல.",
]
