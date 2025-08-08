import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.ocr.ocr_query_service import OCRQueryService
from services.recommendation_service.product_recommendation_service import ProductRecommendationService

class TestOCRQueryService(unittest.TestCase):
    """Test suite for the OCRQueryService."""

    @patch('pipelines.ocr_inference_pipeline.OcrInferencePipeline')
    def setUp(self, MockOcrPipeline):
        """Set up a reusable OCRQueryService instance with mocked dependencies."""
        self.mock_recommendation_service = MagicMock(spec=ProductRecommendationService)
        self.service = OCRQueryService(self.mock_recommendation_service)
        # Keep a reference to the mocked pipeline instance
        self.mock_pipeline_instance = self.service.pipeline

    def test_initialization(self):
        """Tests that the service initializes the OcrInferencePipeline correctly."""
        self.assertIsNotNone(self.service.pipeline)
        # Check that the pipeline was instantiated with the recommendation service
        self.service.pipeline.__class__.assert_called_with(self.mock_recommendation_service)

    def test_process_image_query_success(self):
        """Tests a successful image query processing call."""
        expected_result = {'products': ['laptop'], 'extracted_text': 'laptop'}
        self.mock_pipeline_instance.run.return_value = expected_result

        result = self.service.process_image_query('dummy_image_data')

        self.assertEqual(result, expected_result)
        self.mock_pipeline_instance.run.assert_called_once_with('dummy_image_data')

    def test_process_image_query_failure(self):
        """Tests the error handling when the pipeline fails."""
        self.mock_pipeline_instance.run.side_effect = Exception("Pipeline failure")

        result = self.service.process_image_query('dummy_image_data')

        self.assertIn('error', result)
        self.assertEqual(result['products'], [])
        self.assertEqual(result['extracted_text'], "")
        self.assertIn('An error occurred', result['response'])

    def test_validate_image_format_supported(self):
        """Tests that supported image formats are validated correctly."""
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            with self.subTest(ext=ext):
                mock_file = MagicMock()
                mock_file.filename = f'test_image{ext}'
                self.assertTrue(self.service.validate_image_format(mock_file))

    def test_validate_image_format_unsupported(self):
        """Tests that unsupported image formats are rejected."""
        mock_file = MagicMock()
        mock_file.filename = 'document.pdf'
        self.assertFalse(self.service.validate_image_format(mock_file))

    def test_validate_image_format_no_filename(self):
        """Tests validation when the file object has no filename attribute."""
        # Should default to True as it cannot determine the type
        self.assertTrue(self.service.validate_image_format(b'some_image_bytes'))

if __name__ == '__main__':
    unittest.main()
