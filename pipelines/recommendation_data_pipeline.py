# pipelines/recommendation_data_pipeline.py

import os
import sys

import pandas as pd

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.vector_knowledge_base.embedding_service import EmbeddingService
from services.vector_knowledge_base.vector_service import VectorService


class RecommendationDataPipeline:
    """Orchestrates the data processing for the recommendation service."""

    def __init__(self):
        """Initializes the pipeline with embedding and vector services."""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorService()

    def run(self, data_path='data/dataset/cleaned_dataset.csv'):
        """Executes the full data processing pipeline."""
        print("Starting recommendation data pipeline...")

        # 1. Load data
        df = pd.read_csv(data_path)

        # 2. Clean and filter data
        df = df[
            (df['Description'].str.lower() != 'ebay') &
            (df['UnitPrice'] > 0) &
            (df['Description'].notna()) &
            (df['Description'].str.strip() != '') &
            (df['StockCode'].notna())
        ]

        # 3. Get unique products
        products_df = df.groupby('StockCode').agg({
            'Description': 'first',
            'UnitPrice': 'mean'
        }).reset_index()

        # 4. Generate embeddings
        descriptions = products_df['Description'].tolist()
        embeddings = self.embedding_service.create_embeddings_batch(descriptions)

        # 5. Prepare metadata
        metadata = []
        for _, row in products_df.iterrows():
            if (
                pd.notna(row['Description']) and
                str(row['Description']).strip() != '' and
                str(row['Description']).lower() != 'ebay' and
                pd.notna(row['UnitPrice']) and
                float(row['UnitPrice']) > 0
            ):
                metadata.append({
                    'stock_code': str(row['StockCode']),
                    'description': str(row['Description']),
                    'unit_price': float(row['UnitPrice'])
                })

        # 6. Upsert vectors to the vector store
        self.vector_store.upsert_vectors(embeddings.tolist(), metadata)

        print("Recommendation data pipeline completed successfully!")


if __name__ == '__main__':
    pipeline = RecommendationDataPipeline()
    pipeline.run()
