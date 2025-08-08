import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the service to be tested
from services.recommendation_service.product_recommendation_service import ProductRecommendationService

class TestProductRecommendationService(unittest.TestCase):
    """Test suite for the ProductRecommendationService."""

    @patch('services.recommendation_service.product_recommendation_service.VectorService')
    def setUp(self, MockVectorService):
        """Set up a reusable service instance with a mocked VectorService."""
        self.mock_vector_service_instance = MockVectorService.return_value
        self.service = ProductRecommendationService()

    def test_get_recommendations_success(self):
        """Tests successful retrieval of product recommendations."""
        mock_results = [
            {'metadata': {'stock_code': '123', 'description': 'Test Product 1', 'unit_price': 10.0}},
            {'metadata': {'stock_code': '456', 'description': 'Test Product 2', 'unit_price': 20.0}}
        ]
        self.mock_vector_service_instance.search_by_text.return_value = mock_results

        products, response = self.service.get_recommendations("test query")

        self.assertEqual(len(products), 2)
        self.assertEqual(products[0]['description'], 'Test Product 1')
        self.assertIn("I found 2 products", response)
        self.mock_vector_service_instance.search_by_text.assert_called_once()

    def test_get_recommendations_no_results(self):
        """Tests the case where no products match the query."""
        self.mock_vector_service_instance.search_by_text.return_value = []

        products, response = self.service.get_recommendations("unknown query")

        self.assertEqual(len(products), 0)
        self.assertIn("I couldn't find any products", response)

    def test_get_recommendations_with_price_filter(self):
        """Tests that the price range filter is correctly applied."""
        self.mock_vector_service_instance.search_by_text.return_value = []
        
        price_range = (10.0, 50.0)
        products, response = self.service.get_recommendations("test query", price_range=price_range)

        # Note: The currency symbol might be problematic in some environments, so we check for the core text.
        self.assertIn("price range", response)
        self.assertIn("10.00", response)
        self.assertIn("50.00", response)
        
        # Check that the filter_dict was passed correctly to the search method
        call_args = self.mock_vector_service_instance.search_by_text.call_args
        self.assertIn('filter_dict', call_args.kwargs)
        expected_filter = {
            "unit_price": {
                "$gte": 10.0,
                "$lte": 50.0
            }
        }
        self.assertEqual(call_args.kwargs['filter_dict'], expected_filter)

    def test_get_recommendations_exception(self):
        """Tests the exception handling of the service."""
        self.mock_vector_service_instance.search_by_text.side_effect = Exception("Database error")

        with self.assertRaises(Exception) as context:
            self.service.get_recommendations("any query")
        
        self.assertIn("Error getting recommendations: Database error", str(context.exception))

if __name__ == '__main__':
    unittest.main()
