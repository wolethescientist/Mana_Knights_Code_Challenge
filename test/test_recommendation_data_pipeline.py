import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock dependencies before importing the pipeline
mock_embedding_service = MagicMock()
mock_vector_service = MagicMock()

modules = {
    'services.vector_knowledge_base.embedding_service': MagicMock(EmbeddingService=lambda: mock_embedding_service),
    'services.vector_knowledge_base.vector_service': MagicMock(VectorService=lambda: mock_vector_service)
}

@patch.dict(sys.modules, modules)
class TestRecommendationDataPipeline(unittest.TestCase):
    """Test suite for the RecommendationDataPipeline."""

    @patch('pandas.read_csv')
    def test_run_pipeline(self, mock_read_csv):
        """Tests the full run of the recommendation data pipeline."""
        # Late import to ensure mocks are applied
        from pipelines.recommendation_data_pipeline import RecommendationDataPipeline

        # --- Mock setup ---
        # Mock DataFrame
        mock_data = {
            'StockCode': ['1', '2', '2', '3', '4'],
            'Description': ['Product A', 'Product B', 'Product B', 'ebay', ''],
            'UnitPrice': [10.0, 20.0, 21.0, 5.0, 15.0]
        }
        mock_df = pd.DataFrame(mock_data)
        mock_read_csv.return_value = mock_df

        # Mock embedding service
        mock_embeddings = np.random.rand(2, 10) # 2 valid products
        mock_embedding_service.create_embeddings_batch.return_value = mock_embeddings

        # --- Execution ---
        pipeline = RecommendationDataPipeline()
        pipeline.run('dummy_path.csv')

        # --- Assertions ---
        # 1. Load data
        mock_read_csv.assert_called_once_with('dummy_path.csv')
        
        # 2. Generate embeddings
        # The pipeline should filter down to 2 unique, valid products ('Product A', 'Product B')
        mock_embedding_service.create_embeddings_batch.assert_called_once()
        # Check that the descriptions passed to the embedding service are correct
        call_args = mock_embedding_service.create_embeddings_batch.call_args[0][0]
        self.assertEqual(call_args, ['Product A', 'Product B'])

        # 3. Upsert vectors
        mock_vector_service.upsert_vectors.assert_called_once()
        # Check that the correct number of vectors and metadata entries were passed
        upsert_args = mock_vector_service.upsert_vectors.call_args[0]
        self.assertEqual(len(upsert_args[0]), 2) # embeddings
        self.assertEqual(len(upsert_args[1]), 2) # metadata

        # Check the content of the metadata
        metadata = upsert_args[1]
        self.assertEqual(metadata[0]['stock_code'], '1')
        self.assertEqual(metadata[1]['stock_code'], '2')
        self.assertAlmostEqual(metadata[1]['unit_price'], 20.5) # Mean of 20.0 and 21.0

if __name__ == '__main__':
    unittest.main()
