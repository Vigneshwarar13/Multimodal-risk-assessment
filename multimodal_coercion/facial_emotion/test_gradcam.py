import unittest
import numpy as np
from multimodal_coercion.facial_emotion.tf_emotion_model import EmotionModel, build_cnn
from multimodal_coercion.facial_emotion.pipeline import infer_emotion_on_frame


class TestEmotionModelGradCAM(unittest.TestCase):
    def setUp(self):
        self.model = EmotionModel(model_dir=None)
        # use untrained built model for deterministic behavior
        self.model.model = build_cnn(input_shape=(48, 48, 1), num_classes=5)

    def test_compute_gradcam_shape_and_range(self):
        face_input = np.random.rand(1, 48, 48, 1).astype('float32')
        heatmap = self.model.compute_gradcam(face_input)

        self.assertEqual(heatmap.shape, (48, 48))
        self.assertTrue(np.nanmin(heatmap) >= 0.0)
        self.assertTrue(np.nanmax(heatmap) <= 1.0)

    def test_compute_gradcam_class_index_override(self):
        face_input = np.random.rand(1, 48, 48, 1).astype('float32')
        heatmap = self.model.compute_gradcam(face_input, class_index=2)

        self.assertEqual(heatmap.shape, (48, 48))
        self.assertTrue(np.nanmin(heatmap) >= 0.0)
        self.assertTrue(np.nanmax(heatmap) <= 1.0)

    def test_compute_gradcam_invalid_input(self):
        with self.assertRaises(ValueError):
            self.model.compute_gradcam(np.zeros((48, 48, 1), dtype='float32'))


class TestPipelineGradCAM(unittest.TestCase):
    def test_infer_emotion_on_frame_no_face(self):
        # Black image with no detectable face returns empty results
        black_frame = np.zeros((48, 48, 3), dtype='uint8')
        result = infer_emotion_on_frame(black_frame, return_gradcam=True)

        self.assertEqual(result['emotions'], {})
        self.assertEqual(result['stress_metrics']['stress_prob'], 0.0)
        self.assertEqual(result['stress_metrics']['fear_prob'], 0.0)
        self.assertIsNone(result['gradcam'])


if __name__ == '__main__':
    unittest.main()
