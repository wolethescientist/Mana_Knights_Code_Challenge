import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import logging

# Adjust imports for the new service-oriented structure
from .embedding_service import EmbeddingService
from .similarity_metrics import SimilarityMetricsService

# Load environment variables
load_dotenv()

class VectorService:
    """
    Consolidated service for Pinecone vector database operations.
    Handles index management, vector storage, and multi-metric similarity search.
    """

    def __init__(self, index_name: str = "product-recommendations", metric: str = 'cosine'):
        """
        Initialize the vector database service.

        Args:
            index_name (str): Name of the Pinecone index.
            metric (str): Similarity metric ('cosine', 'euclidean', 'dotproduct').
        """
        self.logger = logging.getLogger(__name__)

        self.index_name = index_name
        self.metric = metric
        self.api_key = os.getenv('PINECONE_API_KEY')

        if not self.api_key:
            self.logger.error("PINECONE_API_KEY not found in environment variables")
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        valid_metrics = ['cosine', 'euclidean', 'dotproduct']
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")

        self.pc = Pinecone(api_key=self.api_key)
        self.embedding_service = EmbeddingService('all-MiniLM-L6-v2')
        self.similarity_service = SimilarityMetricsService()
        self.index = None
        self._setup_index()

    def _setup_index(self):
        """
        Set up the Pinecone index, creating it if it doesn't exist.
        """
        try:
            indexes = self.pc.list_indexes()
            if self.index_name not in indexes.names():
                self.logger.info(f"Creating new index: {self.index_name}")
                dimension = self.embedding_service.get_embedding_dimension()
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=self.metric,
                    spec=spec
                )
                self.logger.info("Index created successfully.")
            else:
                self.logger.info(f"Using existing index: {self.index_name}")

            self.index = self.pc.Index(self.index_name)
            self.logger.info("Vector service initialized successfully.")

        except Exception as e:
            self.logger.error(f"Error setting up Pinecone index: {e}")
            raise

    def upsert_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """
        Insert or update vectors in the database.

        Args:
            vectors (list): List of vectors to insert/update.
            metadata (list): List of metadata dictionaries for each vector.
            ids (list, optional): List of unique IDs for each vector. If None, generated automatically.
        """
        if not vectors or not metadata:
            raise ValueError("Vectors and metadata cannot be empty.")
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries.")

        if ids is None:
            ids = [f"vec_{i}" for i in range(len(vectors))]
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs.")

        try:
            vectors_list = list(zip(ids, vectors, metadata))
            batch_size = 100
            for i in range(0, len(vectors_list), batch_size):
                batch = vectors_list[i:i + batch_size]
                self.index.upsert(vectors=batch)
                self.logger.info(f"Upserted batch of {len(batch)} vectors.")
        except Exception as e:
            self.logger.error(f"Failed to upsert vectors: {e}")
            raise

    def search_by_text(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar products using a text query.
        """
        try:
            query_vector = self.embedding_service.create_embedding(query)
            return self.search_by_vector(query_vector.tolist(), top_k, filter_dict, score_threshold)
        except Exception as e:
            self.logger.error(f"Error in text search: {e}")
            return []

    def search_by_vector(self, query_vector: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors with optional metadata filtering.
        """
        try:
            search_results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            results = []
            for match in search_results.matches:
                info = {'id': match.id, 'score': match.score, 'metadata': match.metadata}
                if score_threshold is not None:
                    if self.metric == 'euclidean':
                        if info['score'] <= score_threshold:
                            results.append(info)
                    elif info['score'] >= score_threshold:
                        results.append(info)
                else:
                    results.append(info)
            return results
        except Exception as e:
            self.logger.error(f"Failed to search by vector: {e}")
            return []

    def load_products(self, dataset_path: str):
        """
        Load products from a dataset into the vector database.
        """
        try:
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded dataset with {len(df)} products.")

            vectors, metadata, ids = [], [], []
            for idx, row in df.iterrows():
                product_data = row.to_dict()
                product_text = self._prepare_product_text(product_data)
                if not product_text.strip():
                    continue

                vector = self.embedding_service.create_embedding(product_text)
                meta = {key: str(val) for key, val in product_data.items() if isinstance(val, (str, int, float))}
                meta['text'] = product_text[:1000]

                vectors.append(vector.tolist())
                metadata.append(meta)
                ids.append(str(product_data.get('id', idx)))

            self.upsert_vectors(vectors, metadata, ids)
            self.logger.info("Successfully loaded all products to vector database.")
        except Exception as e:
            self.logger.error(f"Error loading products to vector database: {e}")
            raise

    def _prepare_product_text(self, product_data: Dict[str, Any]) -> str:
        """
        Prepare product text for vectorization by combining relevant fields.
        """
        fields = ['name', 'description', 'category', 'brand', 'tags']
        return ' '.join(str(product_data.get(f, '')) for f in fields)

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database index.
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return {}

    def delete_index(self):
        """
        Delete the Pinecone index (use with caution).
        """
        try:
            self.pc.delete_index(self.index_name)
            self.logger.info(f"Deleted index: {self.index_name}")
        except Exception as e:
            self.logger.error(f"Error deleting index: {e}")
            raise
