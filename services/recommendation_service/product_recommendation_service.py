"""
Product recommendation service using vector search.

This module defines the `ProductRecommendationService`, which provides product
recommendations based on textual similarity. It uses an `EmbeddingService` to
generate vector embeddings for product descriptions and a `VectorService` to
perform similarity searches.
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Try to import embedding service, make it optional
try:
    from ..vector_knowledge_base.embedding_service import EmbeddingService
    EMBEDDING_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EmbeddingService not available: {e}")
    EMBEDDING_SERVICE_AVAILABLE = False
    EmbeddingService = None
from ..vector_knowledge_base.vector_service import VectorService

class ProductRecommendationService:
    """Service for providing product recommendations.

    This service uses a vector store to find products that are semantically similar
    to a given query. It assumes that the vector store has already been populated
    by the RecommendationDataPipeline.

    Attributes:
        vector_store (VectorService): The service used to store and search vectors.
    """
    def __init__(self) -> None:
        """Initializes the ProductRecommendationService."""
        print("Initializing recommendation service...")
        self.vector_store = VectorService()
        print("Vector store initialized")
    
    def get_recommendations(
        self, 
        query: str, 
        top_k: Optional[int] = None, 
        min_score: float = 0.35, 
        price_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Gets product recommendations based on a natural language query.

        Args:
            query: The natural language query from the user.
            top_k: The maximum number of recommendations to return.
            min_score: The minimum similarity score for a product to be included.
            price_range: An optional tuple (min_price, max_price) for filtering.

        Returns:
            A tuple containing a list of product dictionaries and a natural
            language response string.

        Raises:
            Exception: If an error occurs during the recommendation process.
        """
        try:
            # Prepare metadata filter if price range is provided
            filter_dict = None
            if price_range:
                min_price, max_price = price_range
                filter_dict = {
                    "unit_price": {
                        "$gte": float(min_price),
                        "$lte": float(max_price)
                    }
                }
            
            # Use a reasonable default for top_k if not specified, but allow flexibility
            search_top_k = top_k if top_k is not None else 10000  # Return all available results

            # Search for similar products using the new text-based search method
            results = self.vector_store.search_by_text(
                query=query,
                top_k=search_top_k,
                filter_dict=filter_dict,
                score_threshold=min_score
            )
            
            # Format results - only include product details, no similarity scores
            products = []
            for result in results:
                products.append({
                    'stock_code': result['metadata']['stock_code'],
                    'description': result['metadata']['description'],
                    'unit_price': result['metadata']['unit_price']
                })
            
            # Generate natural language response
            if products:
                response = f"I found {len(products)} products that match your query."
                
                if price_range:
                    response += f" Results are filtered to price range: £{min_price:.2f} - £{max_price:.2f}."
            else:
                response = "I couldn't find any products matching your query"
                if price_range:
                    response += f" in the price range £{min_price:.2f} - £{max_price:.2f}"
                response += ". Please try different search terms or adjust your filters."
            
            return products, response
            
        except Exception as e:
            raise Exception(f"Error getting recommendations: {str(e)}") 