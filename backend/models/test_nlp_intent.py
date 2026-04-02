import unittest
from backend.models.nlp_intent import (analyze_intent_score, detect_dialect_hint)


class TestNLPIntentDialectFallback(unittest.TestCase):
    def test_detect_dialect_hint_known(self):
        text = "அரு வீட்டுல வச்சிருப்போம்"
        hint = detect_dialect_hint(text)
        self.assertIn("Dialectal", hint)

    def test_detect_dialect_hint_unknown(self):
        text = "என்னோ வழக்கு என்பதெல்லாம்"
        hint = detect_dialect_hint(text)
        self.assertIn("Unknown", hint)

    def test_analyze_intent_score_returns_tuple(self):
        text = "நான் சம்மதிக்கிறேன்"
        score, sentiment, coer_flag, nlp_confidence, dialect_fallback, dialect_hint = analyze_intent_score(text)

        self.assertIsInstance(score, float)
        self.assertIsInstance(sentiment, str)
        self.assertIsInstance(coer_flag, bool)
        self.assertIsInstance(nlp_confidence, float)
        self.assertIsInstance(dialect_fallback, bool)
        self.assertIsInstance(dialect_hint, str)

    def test_analyze_intent_score_fallback_on_low_confidence(self):
        # For empty or gibberish, model might fallback
        score, sentiment, coer_flag, nlp_confidence, dialect_fallback, _ = analyze_intent_score("")
        self.assertTrue(dialect_fallback)
        self.assertLessEqual(nlp_confidence, 1.0)
        self.assertGreaterEqual(nlp_confidence, 0.0)


if __name__ == '__main__':
    unittest.main()
