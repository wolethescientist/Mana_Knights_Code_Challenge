"""
Embedding Service for generating text embeddings using sentence transformers.
Provides semantic embeddings for product descriptions and search queries.
"""

import numpy as np
import re
from typing import List, Union, Optional
import logging

try:
    import torch
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Using fallback embeddings.")

class EmbeddingService:
    """
    Service for generating embeddings from text using sentence transformers.
    Handles text preprocessing, validation, and batch processing.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding service.

        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default dimension for all-MiniLM-L6-v2
        self.logger = logging.getLogger(__name__)

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Embedding service initialized with model: {model_name}")
                self.logger.info(f"Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                self.logger.warning(f"Could not load embedding model: {e}")
                self.model = None
        else:
            self.logger.warning("Using fallback embedding service (random vectors)")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding generation.

        Args:
            text (str): Raw input text

        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-\.]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def validate_text(self, text: str) -> bool:
        """
        Validate if text is suitable for embedding generation.

        Args:
            text (str): Text to validate

        Returns:
            bool: True if text is valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False

        # Check if text is not empty after preprocessing
        processed = self.preprocess_text(text)
        if not processed or len(processed.strip()) == 0:
            return False

        # Check if text is not too long (most models have token limits)
        if len(processed) > 5000:  # Conservative limit
            return False

        return True

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text (str): Input text

        Returns:
            np.ndarray: Text embedding vector
        """
        if not self.validate_text(text):
            self.logger.warning(f"Invalid text for embedding: {text[:50]}...")
            return np.random.rand(self.embedding_dim)

        processed_text = self.preprocess_text(text)

        if self.model:
            try:
                embedding = self.model.encode(processed_text)
                return embedding
            except Exception as e:
                self.logger.error(f"Error generating embedding: {e}")
                return np.random.rand(self.embedding_dim)
        else:
            # Fallback: return deterministic random vector based on text hash
            np.random.seed(hash(processed_text) % (2**32))
            return np.random.rand(self.embedding_dim)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts (list): List of input texts

        Returns:
            np.ndarray: Array of text embedding vectors
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        # Validate and preprocess all texts
        processed_texts = []
        for text in texts:
            if self.validate_text(text):
                processed_texts.append(self.preprocess_text(text))
            else:
                processed_texts.append("")  # Empty string for invalid texts

        if self.model:
            try:
                embeddings = self.model.encode(processed_texts)
                return embeddings
            except Exception as e:
                self.logger.error(f"Error generating embeddings: {e}")
                return np.random.rand(len(texts), self.embedding_dim)
        else:
            # Fallback: return deterministic random vectors
            embeddings = []
            for text in processed_texts:
                if text:
                    np.random.seed(hash(text) % (2**32))
                    embeddings.append(np.random.rand(self.embedding_dim))
                else:
                    embeddings.append(np.random.rand(self.embedding_dim))
            return np.array(embeddings)

    def create_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts in batches to handle memory efficiently.

        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing

        Returns:
            np.ndarray: Array of text embedding vectors
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        # Process in batches to avoid memory issues
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings(batch)
            all_embeddings.append(batch_embeddings)

            # Log progress for large batches
            if len(texts) > 100:
                progress = min(i + batch_size, len(texts))
                self.logger.info(f"Processed {progress}/{len(texts)} embeddings...")

        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([]).reshape(0, self.embedding_dim)

    def create_embedding(self, text: str) -> np.ndarray:
        """
        Alias for get_embedding to match the interface expected by vector_database.py

        Args:
            text (str): Input text

        Returns:
            np.ndarray: Text embedding vector
        """
        return self.get_embedding(text)

    def get_model_info(self) -> dict:
        """
        Get information about the current embedding model.

        Returns:
            dict: Model information including name, dimension, and availability
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'model_available': self.model is not None,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }
