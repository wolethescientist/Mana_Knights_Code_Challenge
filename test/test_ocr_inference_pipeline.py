import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock dependencies before importing the pipeline
mock_ocr_service = MagicMock()
mock_recommendation_service = MagicMock()

modules = {
    'services.ocr.ocr_service': MagicMock(OCRService=lambda: mock_ocr_service),
    'services.recommendation_service.product_recommendation_service': MagicMock(ProductRecommendationService=mock_recommendation_service)
}

@patch.dict(sys.modules, modules)
class TestOcrInferencePipeline(unittest.TestCase):
    """Test suite for the OcrInferencePipeline."""

    def setUp(self):
        """Reset mocks for each test."""
        mock_ocr_service.reset_mock()
        mock_recommendation_service.reset_mock()
        # Late import to ensure mocks are applied
        from pipelines.ocr_inference_pipeline import OcrInferencePipeline
        self.pipeline = OcrInferencePipeline(mock_recommendation_service)

    def test_run_success(self):
        """Tests a successful run of the OCR inference pipeline."""
        # Mock service responses
        mock_ocr_service.extract_text_from_image.return_value = ('Test Query', 95.0)
        mock_ocr_service.is_text_readable.return_value = True
        mock_recommendation_service.get_recommendations.return_value = ([{'product': 'A'}], "Found 1 product.")

        result = self.pipeline.run('dummy_image_data')

        # Assertions
        self.assertEqual(result['extracted_text'], 'Test Query')
        self.assertEqual(result['query'], 'Test Query')
        self.assertEqual(len(result['products']), 1)
        self.assertIn("From your query 'Test Query'", result['response'])
        mock_ocr_service.extract_text_from_image.assert_called_once_with('dummy_image_data')
        mock_recommendation_service.get_recommendations.assert_called_once_with('Test Query')

    def test_run_no_text_found(self):
        """Tests the case where the OCR service finds no text."""
        mock_ocr_service.extract_text_from_image.return_value = ('', 0.0)

        result = self.pipeline.run('dummy_image_data')

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No text found in image')

    def test_run_text_not_readable(self):
        """Tests the case where the extracted text is not a readable query."""
        mock_ocr_service.extract_text_from_image.return_value = ('gibberish', 80.0)
        mock_ocr_service.is_text_readable.return_value = False

        result = self.pipeline.run('dummy_image_data')

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Text not readable as product query')

    def test_clean_extracted_text(self):
        """Tests the internal text cleaning method."""
        raw_text = "  Here is some | text with ` artifacts ~  "
        cleaned_text = self.pipeline._clean_extracted_text(raw_text)
        self.assertEqual(cleaned_text, "Here is some text with artifacts")

if __name__ == '__main__':
    unittest.main()
