import unittest
from multimodal_coercion.facial_emotion.temporal_consistency import analyze_temporal_consistency


class TestTemporalConsistency(unittest.TestCase):
    def test_analyze_temporal_consistency_empty(self):
        result = analyze_temporal_consistency([], [])
        self.assertEqual(result["status"], "No Data")
        self.assertEqual(result["inconsistency_score"], 0.0)

    def test_analyze_temporal_consistency_consistent(self):
        stress = [0.2, 0.21, 0.19, 0.2, 0.22, 0.2]
        fear = [0.1, 0.12, 0.09, 0.1, 0.11, 0.1]
        result = analyze_temporal_consistency(stress, fear)
        self.assertEqual(result["status"], "Consistent")
        self.assertLess(result["inconsistency_score"], 0.18)

    def test_analyze_temporal_consistency_inconsistent(self):
        stress = [0.1, 0.2, 0.5, 0.8, 0.7, 0.9]
        fear = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = analyze_temporal_consistency(stress, fear)
        self.assertEqual(result["status"], "Inconsistent")
        self.assertGreater(result["inconsistency_score"], 0.15)


if __name__ == '__main__':
    unittest.main()
