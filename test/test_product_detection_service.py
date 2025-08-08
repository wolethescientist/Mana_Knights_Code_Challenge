import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.image_detection_service.product_detection_service import CNNProductDetectionService

class TestCNNProductDetectionService(unittest.TestCase):
    """Test suite for the CNNProductDetectionService."""

    @patch('pipelines.cnn_inference_pipeline.CNNInferencePipeline')
    def test_initialization_success(self, mock_pipeline):
        """Tests that the service initializes the pipeline on creation."""
        service = CNNProductDetectionService()
        self.assertTrue(service.is_model_ready())
        mock_pipeline.assert_called_once()

    @patch('pipelines.cnn_inference_pipeline.CNNInferencePipeline', side_effect=Exception("Failed to load model"))
    def test_initialization_failure(self, mock_pipeline):
        """Tests that the service handles pipeline initialization errors."""
        service = CNNProductDetectionService()
        self.assertFalse(service.is_model_ready())

    @patch('pipelines.cnn_inference_pipeline.CNNInferencePipeline')
    def test_predict_product_success(self, mock_pipeline):
        """Tests that predict_product calls the pipeline's run method."""
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.run.return_value = {'product': 'test'}
        mock_pipeline.return_value = mock_pipeline_instance

        service = CNNProductDetectionService()
        result = service.predict_product('dummy_image.png')

        self.assertEqual(result, {'product': 'test'})
        mock_pipeline_instance.run.assert_called_once_with('dummy_image.png')

    @patch('pipelines.cnn_inference_pipeline.CNNInferencePipeline', side_effect=Exception("Failed to load model"))
    def test_predict_product_model_not_ready(self, mock_pipeline):
        """Tests that predict_product raises an error if the model is not ready."""
        service = CNNProductDetectionService()
        with self.assertRaises(RuntimeError):
            service.predict_product('dummy_image.png')

if __name__ == '__main__':
    unittest.main()
