import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock dependencies before importing the pipeline
mock_keras_models = MagicMock()
mock_config = MagicMock(MODEL_PATH='dummy_model.h5', ENCODER_PATH='dummy_encoder.pkl', IMAGE_WIDTH=100, IMAGE_HEIGHT=100)

modules = {
    'keras.models': mock_keras_models,
    'cnn_training.config': mock_config
}

@patch.dict(sys.modules, modules)
class TestCNNInferencePipeline(unittest.TestCase):
    """Test suite for the CNNInferencePipeline."""

    @patch('builtins.open', new_callable=mock_open, read_data=b'data')
    @patch('pickle.load')
    def test_run_pipeline(self, mock_pickle_load, mock_file_open):
        """Tests the full run of the CNN inference pipeline."""
        # Late import to ensure mocks are applied
        from pipelines.cnn_inference_pipeline import CNNInferencePipeline

        # --- Mock setup ---
        # Mock Keras model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]]) # Prediction probabilities
        mock_keras_models.load_model.return_value = mock_model

        # Mock Label Encoder
        mock_encoder = MagicMock()
        mock_encoder.classes_ = ['class_a', 'class_b', 'class_c']
        mock_pickle_load.return_value = mock_encoder

        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image # Ensure convert returns the same mock
        with patch('PIL.Image.open', return_value=mock_image):
            # --- Execution ---
            pipeline = CNNInferencePipeline()
            result = pipeline.run('dummy_image_data.png')

        # --- Assertions ---
        # Check model loading
        mock_keras_models.load_model.assert_called_with('dummy_model.h5')
        mock_pickle_load.assert_called_once()

        # Check image preprocessing
        mock_image.convert.assert_called_with('RGB')
        mock_image.resize.assert_called_with((100, 100))

        # Check prediction result
        self.assertEqual(result['predicted_class'], 'class_b')
        self.assertAlmostEqual(result['confidence'], 0.8)
        self.assertEqual(len(result['top_3_predictions']), 3)
        self.assertEqual(result['top_3_predictions'][0]['class'], 'class_b')

if __name__ == '__main__':
    unittest.main()
