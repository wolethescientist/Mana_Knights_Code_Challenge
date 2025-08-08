import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the service to be tested
from services.vector_knowledge_base.embedding_service import EmbeddingService

class TestEmbeddingService(unittest.TestCase):
    """Test suite for the EmbeddingService."""

    def test_preprocess_text(self):
        """Tests the text preprocessing logic."""
        service = EmbeddingService()
        self.assertEqual(service.preprocess_text("  SOME Text with    extra spaces!! "), "some text with extra spaces")
        self.assertEqual(service.preprocess_text(123), "123")

    def test_validate_text(self):
        """Tests the text validation logic."""
        service = EmbeddingService()
        self.assertTrue(service.validate_text("Valid text."))
        self.assertFalse(service.validate_text(""))
        self.assertFalse(service.validate_text(None))
        self.assertFalse(service.validate_text("   "))

    @patch('services.vector_knowledge_base.embedding_service.SentenceTransformer')
    def test_get_embedding_with_model(self, MockSentenceTransformer):
        """Tests single embedding generation when the model is available."""
        mock_model_instance = MockSentenceTransformer.return_value
        mock_model_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_model_instance.get_sentence_embedding_dimension.return_value = 3

        service = EmbeddingService()
        embedding = service.get_embedding("some text")

        self.assertEqual(embedding.shape, (3,))
        mock_model_instance.encode.assert_called_once()

    @patch('services.vector_knowledge_base.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_get_embedding_fallback(self):
        """Tests single embedding generation using the fallback mechanism."""
        service = EmbeddingService()
        embedding = service.get_embedding("some text")
        self.assertEqual(embedding.shape, (service.embedding_dim,))

    @patch('services.vector_knowledge_base.embedding_service.SentenceTransformer')
    def test_get_embeddings_batch(self, MockSentenceTransformer):
        """Tests batch embedding generation."""
        mock_model_instance = MockSentenceTransformer.return_value
        mock_model_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model_instance.get_sentence_embedding_dimension.return_value = 2

        service = EmbeddingService()
        texts = ["text one", "text two"]
        embeddings = service.get_embeddings(texts)

        self.assertEqual(embeddings.shape, (2, 2))
        mock_model_instance.encode.assert_called_once()

    def test_create_embeddings_batch_processing(self):
        """Tests the batch processing wrapper."""
        service = EmbeddingService()
        texts = ["text one", "text two", "text three"]
        
        # Mock the get_embeddings method to check if it's called correctly
        with patch.object(service, 'get_embeddings', return_value=np.random.rand(2, service.embedding_dim)) as mock_get_embeddings:
            service.create_embeddings_batch(texts, batch_size=2)
            # Should be called twice: once for the batch of 2, once for the remaining 1
            self.assertEqual(mock_get_embeddings.call_count, 2)

    def test_get_model_info(self):
        """Tests the model information retrieval method."""
        with patch('services.vector_knowledge_base.embedding_service.SentenceTransformer') as MockSentenceTransformer:
            # Simulate model is available
            MockSentenceTransformer.return_value = MagicMock()
            service_with_model = EmbeddingService()
            info = service_with_model.get_model_info()
            self.assertTrue(info['model_available'])

        with patch('services.vector_knowledge_base.embedding_service.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            # Simulate model is not available
            service_no_model = EmbeddingService()
            info = service_no_model.get_model_info()
            self.assertFalse(info['model_available'])

if __name__ == '__main__':
    unittest.main()
