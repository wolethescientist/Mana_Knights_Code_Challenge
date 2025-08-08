import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.ocr.ocr_service import OCRService, preprocess_image, extract_text

class TestOCRService(unittest.TestCase):
    """Test suite for the OCRService."""

    def setUp(self):
        """Set up a reusable OCRService instance."""
        self.service = OCRService()

    def _create_dummy_image(self, size=(100, 50), color='black') -> Image.Image:
        """Helper to create a dummy PIL image."""
        return Image.new('RGB', size, color)

    @patch('services.ocr.ocr_service.cv2.imread')
    @patch('services.ocr.ocr_service.pytesseract.image_to_string')
    def test_extract_text_from_image_success(self, mock_image_to_string, mock_imread):
        """Tests successful text extraction from an image path."""
        dummy_img_array = np.zeros((50, 100, 3), dtype=np.uint8)
        mock_imread.return_value = dummy_img_array
        mock_image_to_string.return_value = 'Hello World'

        text, confidence = self.service.extract_text_from_image('dummy_path.png')

        self.assertEqual(text, 'Hello World')
        self.assertGreater(confidence, 0)
        mock_imread.assert_called_with('dummy_path.png')

    @patch('services.ocr.ocr_service.cv2.imread', return_value=None)
    def test_extract_text_from_image_invalid_path(self, mock_imread):
        """Tests that an error is raised for an invalid image path."""
        with self.assertRaises(ValueError):
            self.service.extract_text_from_image('invalid_path.png')

    def test_select_best_result(self):
        """Tests the logic for selecting the best OCR result."""
        results = [
            ('method1', 'Short', 60.0),
            ('method2', 'A much longer and more complete sentence.', 95.0),
            ('method3', 'W1th err@rs', 80.0)
        ]
        best_text, best_confidence = self.service._select_best_result(results)
        self.assertEqual(best_text, 'A much longer and more complete sentence.')
        self.assertEqual(best_confidence, 95.0)

    def test_select_best_result_empty(self):
        """Tests that empty results return an empty string."""
        best_text, best_confidence = self.service._select_best_result([])
        self.assertEqual(best_text, "")
        self.assertEqual(best_confidence, 0.0)

    @patch('services.ocr.ocr_service.pytesseract.image_to_string', return_value='Test')
    def test_extract_with_tesseract(self, mock_image_to_string):
        """Tests the internal Tesseract extraction wrapper."""
        dummy_image = self._create_dummy_image()
        text, conf = self.service._extract_with_tesseract(dummy_image, 'test_method')
        self.assertEqual(text, 'Test')
        self.assertGreater(conf, 0)

# --- Standalone Function Tests ---
@patch('services.ocr.ocr_service.OCRService')
class TestStandaloneFunctions(unittest.TestCase):
    """Test suite for standalone OCR functions."""

    def test_preprocess_image_standalone(self, MockOCRService):
        """Tests the standalone preprocess_image function."""
        mock_service_instance = MockOCRService.return_value
        mock_service_instance.preprocess_image.return_value = np.array([1, 2, 3])

        # Mock cv2.imread since it's called inside the function
        with patch('services.ocr.ocr_service.cv2.imread', return_value=np.array([1])):
            result = preprocess_image('dummy_path.png')

        self.assertTrue(np.array_equal(result, np.array([1, 2, 3])))
        mock_service_instance.preprocess_image.assert_called_once()

    def test_extract_text_standalone(self, MockOCRService):
        """Tests the standalone extract_text function."""
        mock_service_instance = MockOCRService.return_value
        mock_service_instance.extract_text_from_image.return_value = ('Success', 99.0)

        result = extract_text('dummy_path.png')

        self.assertEqual(result, 'Success')
        mock_service_instance.extract_text_from_image.assert_called_once_with('dummy_path.png')

    def test_extract_text_standalone_no_text(self, MockOCRService):
        """Tests the standalone extract_text function when no text is found."""
        mock_service_instance = MockOCRService.return_value
        mock_service_instance.extract_text_from_image.return_value = ('', 10.0)

        result = extract_text('dummy_path.png')

        self.assertIn('No text found', result)

if __name__ == '__main__':
    unittest.main()
