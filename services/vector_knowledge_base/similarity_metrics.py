import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances
import logging

class SimilarityMetricsService:
    """
    Service for evaluating and comparing different similarity metrics for product vectors.
    Provides analysis and justification for metric selection.
    """
    
    def __init__(self):
        """
        Initialize the similarity metrics service.
        """
        self.logger = logging.getLogger(__name__)
        
        # Define available similarity metrics
        self.metrics = {
            'cosine': self._cosine_similarity,
            'dot_product': self._dot_product,
            'euclidean': self._euclidean_similarity,
            'manhattan': self._manhattan_similarity
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Input vectors
            
        Returns:
            float: Cosine similarity score (0 to 1)
        """
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def _dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate dot product between two vectors.
        
        Args:
            vec1, vec2: Input vectors
            
        Returns:
            float: Dot product score
        """
        return np.dot(vec1, vec2)
    
    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate similarity based on negative euclidean distance.
        
        Args:
            vec1, vec2: Input vectors
            
        Returns:
            float: Similarity score (higher = more similar)
        """
        distance = np.linalg.norm(vec1 - vec2)
        # Convert distance to similarity (inverse relationship)
        return 1 / (1 + distance)
    
    def _manhattan_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate similarity based on negative manhattan distance.
        
        Args:
            vec1, vec2: Input vectors
            
        Returns:
            float: Similarity score (higher = more similar)
        """
        distance = np.sum(np.abs(vec1 - vec2))
        # Convert distance to similarity (inverse relationship)
        return 1 / (1 + distance)
    
    def evaluate_metrics(self, query_vector: np.ndarray, product_vectors: List[np.ndarray], 
                        product_texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate different similarity metrics on sample product vectors.
        
        Args:
            query_vector: Query vector
            product_vectors: List of product vectors
            product_texts: List of product text descriptions
            
        Returns:
            Dict: Evaluation results for each metric
        """
        results = {}
        
        for metric_name, metric_func in self.metrics.items():
            scores = []
            for i, product_vec in enumerate(product_vectors):
                score = metric_func(query_vector, product_vec)
                scores.append({
                    'index': i,
                    'score': score,
                    'text': product_texts[i][:100] if i < len(product_texts) else "N/A"
                })
            
            # Sort by score (descending for similarity, ascending for distance-based)
            if metric_name in ['euclidean', 'manhattan']:
                scores.sort(key=lambda x: x['score'], reverse=True)
            else:
                scores.sort(key=lambda x: x['score'], reverse=True)
            
            results[metric_name] = {
                'top_5_scores': scores[:5],
                'score_range': {
                    'min': min(s['score'] for s in scores),
                    'max': max(s['score'] for s in scores),
                    'mean': np.mean([s['score'] for s in scores])
                }
            }
        
        return results
    
    def justify_metric_selection(self) -> Dict[str, Any]:
        """
        Provide justification for choosing cosine similarity as the primary metric.
        
        Returns:
            Dict: Justification analysis
        """
        justification = {
            'selected_metric': 'cosine_similarity',
            'reasons': {
                'semantic_understanding': {
                    'description': 'Cosine similarity focuses on the angle between vectors, not magnitude',
                    'benefit': 'Captures semantic similarity regardless of text length or frequency',
                    'example': '"gaming laptop" and "gaming computer" have similar angles despite different word counts'
                },
                'normalization': {
                    'description': 'Automatically normalizes vectors to unit length',
                    'benefit': 'Handles varying text lengths and embedding magnitudes consistently',
                    'example': 'Product descriptions of different lengths are compared fairly'
                },
                'e_commerce_optimized': {
                    'description': 'Ideal for product matching where semantic meaning matters more than exact matches',
                    'benefit': 'Finds related products even with different vocabulary',
                    'example': '"wireless headphones" matches "bluetooth earbuds"'
                },
                'embedding_compatibility': {
                    'description': 'Works optimally with modern embedding models like all-MiniLM-L6-v2',
                    'benefit': 'Leverages the semantic properties of transformer-based embeddings',
                    'example': 'Better semantic understanding than distance-based metrics'
                }
            },
            'comparison_with_alternatives': {
                'dot_product': {
                    'pros': 'Simple computation, preserves magnitude information',
                    'cons': 'Sensitive to vector magnitudes, less suitable for semantic similarity',
                    'verdict': 'Not ideal for semantic product matching'
                },
                'euclidean_distance': {
                    'pros': 'Intuitive distance measure',
                    'cons': 'Sensitive to vector magnitudes, requires normalization',
                    'verdict': 'Good for exact matches, less suitable for semantic similarity'
                },
                'manhattan_distance': {
                    'pros': 'Robust to outliers',
                    'cons': 'Less sensitive to semantic relationships',
                    'verdict': 'Better for categorical data than semantic text'
                }
            },
            'recommendation': {
                'primary': 'cosine_similarity',
                'reasoning': 'Best balance of semantic understanding, normalization, and e-commerce requirements',
                'implementation': 'Already configured in Pinecone index with metric="cosine"'
            }
        }
        
        return justification
    
    def get_metric_characteristics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed characteristics of each similarity metric.
        
        Returns:
            Dict: Characteristics of each metric
        """
        characteristics = {
            'cosine_similarity': {
                'range': '0 to 1',
                'interpretation': '1 = identical, 0 = orthogonal, -1 = opposite',
                'advantages': [
                    'Angle-based, magnitude-invariant',
                    'Natural for semantic similarity',
                    'Works well with normalized embeddings',
                    'Industry standard for text similarity'
                ],
                'disadvantages': [
                    'May miss magnitude differences',
                    'Requires non-zero vectors'
                ],
                'best_for': 'Semantic text similarity, product recommendations'
            },
            'dot_product': {
                'range': 'Unbounded',
                'interpretation': 'Higher = more similar (when vectors are normalized)',
                'advantages': [
                    'Simple computation',
                    'Preserves magnitude information',
                    'Direct geometric interpretation'
                ],
                'disadvantages': [
                    'Sensitive to vector magnitudes',
                    'Requires normalization for fair comparison',
                    'Less intuitive for semantic similarity'
                ],
                'best_for': 'Exact matches, magnitude-sensitive comparisons'
            },
            'euclidean_distance': {
                'range': '0 to infinity',
                'interpretation': '0 = identical, higher = more different',
                'advantages': [
                    'Intuitive distance measure',
                    'Geometric interpretation',
                    'Well-understood properties'
                ],
                'disadvantages': [
                    'Sensitive to vector magnitudes',
                    'Requires normalization',
                    'Less suitable for semantic similarity'
                ],
                'best_for': 'Exact vector matching, geometric similarity'
            },
            'manhattan_distance': {
                'range': '0 to infinity',
                'interpretation': '0 = identical, higher = more different',
                'advantages': [
                    'Robust to outliers',
                    'Computationally simple',
                    'Good for categorical data'
                ],
                'disadvantages': [
                    'Less sensitive to semantic relationships',
                    'May not capture complex text patterns',
                    'Requires normalization'
                ],
                'best_for': 'Categorical data, robust distance measurement'
            }
        }
        
        return characteristics 