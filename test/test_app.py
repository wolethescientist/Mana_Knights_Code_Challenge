import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import io

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock services before importing the app
mock_recommendation_service = MagicMock()
mock_ocr_query_service = MagicMock()
mock_cnn_service = MagicMock()

modules = {
    'services.recommendation_service.product_recommendation_service': MagicMock(ProductRecommendationService=lambda: mock_recommendation_service),
    'services.ocr.ocr_query_service': MagicMock(OCRQueryService=lambda r: mock_ocr_query_service),
    'services.image_detection_service.product_detection_service': MagicMock(CNNProductDetectionService=lambda: mock_cnn_service)
}

with patch.dict(sys.modules, modules):
    from app import app

class TestApp(unittest.TestCase):
    """Test suite for the Flask application endpoints."""

    def setUp(self):
        """Set up the Flask test client and reset mocks."""
        self.app = app.test_client()
        self.app.testing = True
        mock_recommendation_service.reset_mock()
        mock_ocr_query_service.reset_mock()
        mock_cnn_service.reset_mock()

    def test_product_recommendation_success(self):
        """Tests the /product-recommendation endpoint with a valid query."""
        mock_recommendation_service.get_recommendations.return_value = ([], "Found 0 products.")
        response = self.app.post('/product-recommendation', data={'query': 'laptop'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Found 0 products.', response.data)

    def test_product_recommendation_no_query(self):
        """Tests the /product-recommendation endpoint with a missing query."""
        response = self.app.post('/product-recommendation', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Query is required', response.data)

    def test_ocr_query_success(self):
        """Tests the /ocr-query endpoint with a valid image."""
        mock_ocr_query_service.process_image_query.return_value = {'extracted_text': 'test'}
        data = {'image_data': (io.BytesIO(b"dummy image data"), 'test.jpg')}
        response = self.app.post('/ocr-query', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'test', response.data)

    def test_ocr_query_no_file(self):
        """Tests the /ocr-query endpoint with no file."""
        response = self.app.post('/ocr-query', content_type='multipart/form-data', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No image file provided', response.data)

    def test_image_product_search_success(self):
        """Tests the /image-product-search endpoint with a valid image."""
        mock_cnn_service.is_model_ready.return_value = True
        mock_cnn_service.predict_product.return_value = {
            'predicted_class': 'monitor', 'confidence': 0.99, 'top_3_predictions': []
        }
        mock_recommendation_service.get_recommendations.return_value = ([], "No products found.")
        data = {'product_image': (io.BytesIO(b"dummy image data"), 'test.png')}
        response = self.app.post('/image-product-search', content_type='multipart/form-data', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'monitor', response.data)

    def test_page_rendering(self):
        """Tests that all GET routes for pages render successfully."""
        pages = ['/', '/text-search', '/ocr-query', '/image-detection', '/services', '/sample_response']
        for page in pages:
            with self.subTest(page=page):
                response = self.app.get(page)
                self.assertEqual(response.status_code, 200, f"Failed to render {page}")

if __name__ == '__main__':
    unittest.main()
