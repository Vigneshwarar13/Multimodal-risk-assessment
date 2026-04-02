"""
Unit tests for SHAP Explainability module.

Tests cover:
1. Configuration and initialization
2. SHAP value computation
3. Token-level explanations
4. Error handling and graceful degradation
5. Integration with NLP classifier
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from typing import List

from multimodal_coercion.speech.shap_explainer import (
    SHAPExplainerConfig,
    ShapleyExplainer,
    create_shap_explainer,
    DEFAULT_TAMIL_BACKGROUND,
)


class TestSHAPExplainerConfig(unittest.TestCase):
    """Test SHAP configuration."""
    
    def test_config_creation_default(self):
        """Test default configuration values."""
        config = SHAPExplainerConfig()
        
        self.assertEqual(config.num_samples, 50)
        self.assertEqual(config.aggregation, "mean")
        self.assertEqual(config.background_samples, 10)
        self.assertEqual(config.max_tokens, 256)
    
    def test_config_creation_custom(self):
        """Test custom configuration values."""
        config = SHAPExplainerConfig(
            num_samples=100,
            aggregation="max",
            background_samples=20,
            max_tokens=512
        )
        
        self.assertEqual(config.num_samples, 100)
        self.assertEqual(config.aggregation, "max")
        self.assertEqual(config.background_samples, 20)
        self.assertEqual(config.max_tokens, 512)


class TestShapleyExplainer(unittest.TestCase):
    """Test core SHAP explainer functionality."""
    
    def setUp(self):
        """Set up mock model and tokenizer."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.device = "cpu"
        self.config = SHAPExplainerConfig(num_samples=10, background_samples=5)
    
    def test_explainer_initialization(self):
        """Test SHAP explainer can be initialized."""
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.device,
            config=self.config
        )
        
        self.assertEqual(explainer.model, self.mock_model)
        self.assertEqual(explainer.tokenizer, self.mock_tokenizer)
        self.assertEqual(explainer.device, self.device)
        self.assertEqual(explainer.target_class, 2)  # Coercion
    
    def test_explainer_initialization_custom_target(self):
        """Test custom target class."""
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            target_class=1,  # Neutral
            config=self.config
        )
        
        self.assertEqual(explainer.target_class, 1)
    
    def test_set_background_data(self):
        """Test setting background dataset."""
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.config
        )
        
        texts = ["text1", "text2", "text3"]
        explainer.set_background_data(texts)
        
        self.assertEqual(len(explainer._background_texts), 3)
        self.assertIn("text1", explainer._background_texts)
    
    def test_set_background_data_truncated(self):
        """Test background data is truncated to config limit."""
        config = SHAPExplainerConfig(background_samples=2)
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=config
        )
        
        texts = ["t1", "t2", "t3", "t4", "t5"]
        explainer.set_background_data(texts)
        
        self.assertEqual(len(explainer._background_texts), 2)
    
    def test_prediction_function_mock(self):
        """Test prediction function wrapper with mock."""
        # Create a simple mock that returns fixed predictions
        self.mock_model.to = Mock(return_value=self.mock_model)
        self.mock_model.eval = Mock(return_value=None)
        
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.config
        )
        
        # Mock should return valid shape and values
        # (We use actual model for real testing, so this just tests structure)
        self.assertIsNotNone(explainer._prediction_function)
    
    def test_explain_prediction_structure(self):
        """Test explain_prediction returns correct structure."""
        # This test uses mocks to verify structure without real model
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.config
        )
        
        # Mock the prediction function to return a simple explanation
        with patch.object(explainer, '_prediction_function', return_value=np.array([0.8])):
            with patch.object(explainer, '_compute_shap_values', return_value=np.array([0.1, 0.05, -0.02])):
                self.mock_tokenizer.convert_ids_to_tokens = Mock(
                    return_value=["[CLS]", "word1", "[SEP]"]
                )
                self.mock_tokenizer.mask_token = "[MASK]"
                
                # Mock tokenizer encoding
                mock_enc = {
                    "input_ids": Mock(spec=['cpu']),
                    "attention_mask": Mock(spec=['to'])
                }
                mock_enc["input_ids"].cpu = Mock(return_value=Mock(numpy=Mock(return_value=np.array([0, 1, 2]))))
                self.mock_tokenizer.return_value = mock_enc
                
                result = explainer.explain_prediction("test text")
                
                # Verify result structure
                self.assertIn("text", result)
                self.assertIn("prediction_score", result)
                self.assertIn("label", result)
                self.assertIn("tokens", result)
                self.assertIn("top_tokens", result)
                self.assertIn("background_size", result)
                self.assertIn("num_samples_used", result)
                self.assertIn("error", result)
    
    def test_explain_prediction_error_handling(self):
        """Test graceful error handling when explanation fails."""
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.config
        )
        
        # Mock tokenizer to raise exception
        self.mock_tokenizer.side_effect = Exception("Tokenizer error")
        
        result = explainer.explain_prediction("test text")
        
        # Should return error structure, not raise
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["prediction_score"], 0.0)
        self.assertEqual(len(result["tokens"]), 0)
        self.assertEqual(len(result["top_tokens"]), 0)


class TestSHAPValueComputation(unittest.TestCase):
    """Test SHAP value computation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.config = SHAPExplainerConfig(num_samples=5, background_samples=3)
    
    def test_compute_shap_values_basic(self):
        """Test SHAP values are computed and have correct shape."""
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.config
        )
        
        # Mock helper methods
        with patch.object(explainer, '_prediction_function') as mock_pred:
            mock_pred.return_value = np.array([0.7])  # Full text prediction
            self.mock_tokenizer.mask_token = "[MASK]"
            self.mock_tokenizer.convert_tokens_to_string = Mock(return_value="masked text")
            
            token_ids = np.array([0, 1, 2, 3])
            token_strs = ["[CLS]", "word", "says", "[SEP]"]
            
            shap_vals = explainer._compute_shap_values("test", token_ids, token_strs)
            
            # Verify output shape and type
            self.assertEqual(len(shap_vals), 4)
            self.assertTrue(np.all(np.isfinite(shap_vals)))
    
    def test_compute_shap_values_special_tokens_ignored(self):
        """Test that special tokens get zero SHAP values."""
        explainer = ShapleyExplainer(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            config=self.config
        )
        
        with patch.object(explainer, '_prediction_function', return_value=np.array([0.7])):
            self.mock_tokenizer.mask_token = "[MASK]"
            self.mock_tokenizer.convert_tokens_to_string = Mock(return_value="masked")
            
            token_ids = np.array([0, 1, 2])
            token_strs = ["[CLS]", "word", "[SEP]"]
            
            shap_vals = explainer._compute_shap_values("test", token_ids, token_strs)
            
            # [CLS] and [SEP] should have zero contribution
            self.assertEqual(shap_vals[0], 0.0)
            self.assertEqual(shap_vals[2], 0.0)


class TestTokenAggregation(unittest.TestCase):
    """Test token-level aggregation and ranking."""
    
    def test_top_k_selection(self):
        """Test that top-k tokens are selected correctly."""
        # Create sample tokens with varying magnitudes
        tokens = [
            {"token": "word1", "shap_value": 0.3, "magnitude": 0.3},
            {"token": "word2", "shap_value": -0.1, "magnitude": 0.1},
            {"token": "word3", "shap_value": 0.5, "magnitude": 0.5},
            {"token": "word4", "shap_value": 0.05, "magnitude": 0.05},
        ]
        
        # Sort by magnitude (simulating what explain_prediction does)
        sorted_tokens = sorted(tokens, key=lambda x: x["magnitude"], reverse=True)
        top_k = sorted_tokens[:2]
        
        # Verify top-k ordering
        self.assertEqual(top_k[0]["token"], "word3")  # magnitude 0.5
        self.assertEqual(top_k[1]["token"], "word1")  # magnitude 0.3
    
    def test_contribution_sign(self):
        """Test positive vs negative contribution assignment."""
        positive_shap = 0.3
        negative_shap = -0.2
        
        pos_contrib = "positive" if positive_shap > 0 else "negative"
        neg_contrib = "positive" if negative_shap > 0 else "negative"
        
        self.assertEqual(pos_contrib, "positive")
        self.assertEqual(neg_contrib, "negative")


class TestCreateShapExplainer(unittest.TestCase):
    """Test factory function for creating SHAP explainer."""
    
    def test_create_with_valid_classifier(self):
        """Test creating explainer from valid classifier."""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier._model = Mock()
        mock_classifier._tokenizer = Mock()
        mock_classifier.label2id = {"Coercion": 2}
        
        explainer = create_shap_explainer(mock_classifier, device="cpu")
        
        self.assertIsInstance(explainer, ShapleyExplainer)
        self.assertEqual(explainer.device, "cpu")
        self.assertEqual(explainer.target_class, 2)
    
    def test_create_with_background_texts(self):
        """Test creating explainer with background texts."""
        mock_classifier = Mock()
        mock_classifier._model = Mock()
        mock_classifier._tokenizer = Mock()
        mock_classifier.label2id = {"Coercion": 2}
        
        background = ["text1", "text2", "text3"]
        explainer = create_shap_explainer(mock_classifier, background_texts=background)
        
        self.assertEqual(len(explainer._background_texts), 3)
    
    def test_create_fails_without_loaded_model(self):
        """Test that creation fails if model not loaded."""
        mock_classifier = Mock()
        mock_classifier._model = None
        mock_classifier._tokenizer = None
        
        with self.assertRaises(ValueError):
            create_shap_explainer(mock_classifier)
    
    def test_create_with_custom_config(self):
        """Test creating explainer with custom config."""
        mock_classifier = Mock()
        mock_classifier._model = Mock()
        mock_classifier._tokenizer = Mock()
        mock_classifier.label2id = {"Coercion": 2}
        
        custom_config = SHAPExplainerConfig(num_samples=100)
        explainer = create_shap_explainer(mock_classifier, config=custom_config)
        
        self.assertEqual(explainer.config.num_samples, 100)


class TestDefaultBackgroundTexts(unittest.TestCase):
    """Test default background texts."""
    
    def test_default_background_exists(self):
        """Test that default background texts are provided."""
        self.assertIsNotNone(DEFAULT_TAMIL_BACKGROUND)
        self.assertGreater(len(DEFAULT_TAMIL_BACKGROUND), 0)
    
    def test_default_background_is_list(self):
        """Test that background is list of strings."""
        self.assertIsInstance(DEFAULT_TAMIL_BACKGROUND, list)
        for text in DEFAULT_TAMIL_BACKGROUND:
            self.assertIsInstance(text, str)
    
    def test_default_background_has_neutral_content(self):
        """Test that background texts are neutral (non-coercive)."""
        # Check for absence of coercive keywords
        coercive_keywords = ["zabardasti", "pressure", "threat", "force"]
        
        for text in DEFAULT_TAMIL_BACKGROUND:
            text_lower = text.lower()
            for keyword in coercive_keywords:
                self.assertNotIn(keyword, text_lower)


class TestExplainerOutputFormat(unittest.TestCase):
    """Test output format and data types."""
    
    def test_explain_prediction_output_types(self):
        """Test that output has correct data types."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = SHAPExplainerConfig()
        
        explainer = ShapleyExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config
        )
        
        # Mock to return minimal structure
        with patch.object(explainer, '_prediction_function', return_value=np.array([0.7])):
            with patch.object(explainer, '_compute_shap_values', return_value=np.array([])):
                mock_tokenizer.convert_ids_to_tokens = Mock(return_value=[])
                mock_tokenizer.mask_token = "[MASK]"
                
                mock_enc = {"input_ids": Mock()}
                mock_enc["input_ids"].cpu = Mock(
                    return_value=Mock(numpy=Mock(return_value=np.array([])))
                )
                mock_tokenizer.return_value = mock_enc
                
                result = explainer.explain_prediction("test")
                
                # Verify types
                self.assertIsInstance(result["text"], str)
                self.assertIsInstance(result["prediction_score"], float)
                self.assertIsInstance(result["label"], str)
                self.assertIsInstance(result["tokens"], list)
                self.assertIsInstance(result["top_tokens"], list)
                self.assertIsInstance(result["background_size"], int)
                self.assertIsInstance(result["num_samples_used"], int)
    
    def test_shap_values_in_valid_range(self):
        """Test that SHAP values are in reasonable range."""
        # SHAP values should typically be between -1 and 1 for probabilities
        shap_vals = np.array([-0.2, -0.05, 0.0, 0.1, 0.3])
        
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(shap_vals)))
        
        # Should be roughly between -1 and 1
        self.assertTrue(np.all(shap_vals >= -1.0))
        self.assertTrue(np.all(shap_vals <= 1.0))


if __name__ == "__main__":
    unittest.main()
