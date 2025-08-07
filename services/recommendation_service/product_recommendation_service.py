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

    This service loads product data, generates embeddings for product descriptions,
    and uses a vector store to find products that are semantically similar to a
    given query.

    Attributes:
        embedding_service (EmbeddingService): The service used to create text embeddings.
        vector_store (VectorService): The service used to store and search vectors.
        products_df (pd.DataFrame): A DataFrame containing the unique products.
    """
    def __init__(self) -> None:
        """Initializes the ProductRecommendationService."""
        print("Initializing recommendation service...")

        if EMBEDDING_SERVICE_AVAILABLE:
            self.embedding_service = EmbeddingService()
            print("Embedding service initialized")
        else:
            self.embedding_service = None
            print("Warning: Embedding service not available, using fallback methods")

        self.vector_store = VectorService()
        print("Vector store initialized")

        # Load and prepare product data
        print("Loading product data...")
        self.load_product_data()
        print("Product data loaded")
    
    def load_product_data(self) -> None:
        """Loads product data from a CSV file and populates the vector store.

        This method reads product data, cleans it, generates embeddings for the
        product descriptions, and upserts the vectors into the vector store.

        Raises:
            Exception: If the product data fails to load or process.
        """
        try:
            print("Loading product data from CSV...")
            df = pd.read_csv('data/dataset/cleaned_dataset.csv')
            print(f"Loaded {len(df)} rows from CSV")

            # Filter out invalid entries before grouping
            print("Filtering out invalid entries...")
            # Remove rows with 'Ebay' descriptions, zero prices, or invalid data
            df = df[
                (df['Description'].str.lower() != 'ebay') &
                (df['UnitPrice'] > 0) &
                (df['Description'].notna()) &
                (df['Description'].str.strip() != '') &
                (df['StockCode'].notna())
            ]
            print(f"After filtering: {len(df)} valid rows")

            # Group by StockCode to get unique products
            self.products_df = df.groupby('StockCode').agg({
                'Description': 'first',
                'UnitPrice': 'mean'
            }).reset_index()
            print(f"Found {len(self.products_df)} unique products")
            
            # Generate embeddings for all product descriptions
            print("Generating embeddings for product descriptions...")
            descriptions = self.products_df['Description'].tolist()
            embeddings = self.embedding_service.create_embeddings_batch(descriptions)
            print(f"Generated {len(embeddings)} embeddings")
            
            # Prepare metadata for vector store
            print("Preparing metadata...")
            metadata = []
            for _, row in self.products_df.iterrows():
                # Ensure valid data before adding to metadata
                if (pd.notna(row['Description']) and
                    str(row['Description']).strip() != '' and
                    str(row['Description']).lower() != 'ebay' and
                    pd.notna(row['UnitPrice']) and
                    float(row['UnitPrice']) > 0):

                    metadata.append({
                        'stock_code': str(row['StockCode']),
                        'description': str(row['Description']),
                        'unit_price': float(row['UnitPrice'])
                    })
            print(f"Prepared metadata for {len(metadata)} valid products")
            
            # Upload to vector store
            print("Uploading vectors to Pinecone...")
            self.vector_store.upsert_vectors(embeddings.tolist(), metadata)
            print("Product data loaded successfully")
            
        except Exception as e:
            raise Exception(f"Failed to load product data: {str(e)}")
    
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