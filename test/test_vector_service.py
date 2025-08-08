import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Set a dummy API key for testing
os.environ['PINECONE_API_KEY'] = 'test-key'

# Import the service to be tested
from services.vector_knowledge_base.vector_service import VectorService

# Mock the Pinecone and other dependencies before they are imported by the service
mock_pinecone = MagicMock()
mock_embedding_service = MagicMock()

modules = {
    'pinecone': mock_pinecone,
    'pinecone.grpc': MagicMock(),
    'pinecone.core.grpc.protos': MagicMock(),
    'services.vector_knowledge_base.embedding_service': MagicMock(EmbeddingService=mock_embedding_service),
    'services.vector_knowledge_base.similarity_metrics': MagicMock(),
    'dotenv': MagicMock(load_dotenv=MagicMock())
}

@patch.dict(sys.modules, modules)
class TestVectorService(unittest.TestCase):
    """Test suite for the VectorService."""

    def setUp(self):
        """Set up a reusable VectorService instance with mocked dependencies."""
        # Reset mocks for each test
        mock_pinecone.reset_mock()
        mock_embedding_service.reset_mock()

        # Mock the Pinecone client and index
        self.mock_pc_instance = mock_pinecone.Pinecone.return_value
        self.mock_index = self.mock_pc_instance.Index.return_value
        
        # Mock list_indexes to simulate the index not existing initially
        mock_index_list = MagicMock()
        mock_index_list.names.return_value = []
        self.mock_pc_instance.list_indexes.return_value = mock_index_list
        
        # Mock the embedding service
        self.mock_embedding_instance = mock_embedding_service.return_value
        self.mock_embedding_instance.get_embedding_dimension.return_value = 8
        self.mock_embedding_instance.create_embedding.return_value = [0.1] * 8

        # Instantiate the service
        self.service = VectorService()

    def test_initialization_creates_index(self):
        """Tests that the service creates a new index if it does not exist."""
        self.mock_pc_instance.create_index.assert_called_once()
        self.mock_pc_instance.Index.assert_called_with("product-recommendations")

    def test_upsert_vectors(self):
        """Tests the vector upsert functionality."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [{'product': 'A'}, {'product': 'B'}]
        self.service.upsert_vectors(vectors, metadata)
        self.mock_index.upsert.assert_called_once()

    def test_search_by_text(self):
        """Tests searching by a text query."""
        self.service.search_by_text("test query")
        self.mock_embedding_instance.create_embedding.assert_called_once_with("test query")
        self.mock_index.query.assert_called_once()

    def test_search_by_vector_with_filter_and_threshold(self):
        """Tests searching with metadata filters and a score threshold."""
        mock_match = MagicMock()
        mock_match.id = 'vec1'
        mock_match.score = 0.9
        mock_match.metadata = {'desc': 'match'}
        self.mock_index.query.return_value = MagicMock(matches=[mock_match])

        results = self.service.search_by_vector([0.1]*8, top_k=1, filter_dict={'type': 'A'}, score_threshold=0.8)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 'vec1')
        call_args = self.mock_index.query.call_args.kwargs
        self.assertEqual(call_args['filter'], {'type': 'A'})

    @patch('services.vector_knowledge_base.vector_service.pd.read_csv')
    def test_load_products(self, mock_read_csv):
        """Tests loading products from a CSV file."""
        # Create a mock DataFrame
        mock_df = pd.DataFrame({
            'name': ['Product A'],
            'description': ['A great product'],
            'id': ['prod1']
        })
        mock_read_csv.return_value = mock_df
        
        # Mock the upsert method to prevent actual calls
        with patch.object(self.service, 'upsert_vectors') as mock_upsert:
            self.service.load_products('dummy_path.csv')
            mock_read_csv.assert_called_once_with('dummy_path.csv')
            self.mock_embedding_instance.create_embedding.assert_called_once()
            mock_upsert.assert_called_once()

    def test_get_index_stats(self):
        """Tests retrieval of index statistics."""
        self.service.get_index_stats()
        self.mock_index.describe_index_stats.assert_called_once()

if __name__ == '__main__':
    unittest.main()
