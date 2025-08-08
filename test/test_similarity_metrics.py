import sys
import os
import unittest
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the class to be tested
from services.vector_knowledge_base.similarity_metrics import SimilarityMetrics

class TestSimilarityMetrics(unittest.TestCase):
    """Test suite for the SimilarityMetrics class."""

    def setUp(self):
        """Set up a reusable instance and test vectors."""
        self.metrics = SimilarityMetrics()
        self.vec_a = np.array([1, 2, 3])
        self.vec_b = np.array([4, 5, 6])
        self.vec_c = np.array([1, 2, 3]) # Identical to vec_a
        self.vec_d = np.array([-1, -2, -3]) # Opposite to vec_a
        self.vec_zero = np.array([0, 0, 0])

    def test_cosine_similarity(self):
        """Tests the cosine similarity calculation."""
        # Test identical vectors
        self.assertAlmostEqual(self.metrics.cosine_similarity(self.vec_a, self.vec_c), 1.0)
        # Test different vectors
        self.assertAlmostEqual(self.metrics.cosine_similarity(self.vec_a, self.vec_b), 0.9746318)
        # Test opposite vectors
        self.assertAlmostEqual(self.metrics.cosine_similarity(self.vec_a, self.vec_d), -1.0)
        # Test with zero vector
        self.assertEqual(self.metrics.cosine_similarity(self.vec_a, self.vec_zero), 0.0)

    def test_dot_product(self):
        """Tests the dot product calculation."""
        self.assertEqual(self.metrics.dot_product(self.vec_a, self.vec_b), 32)
        self.assertEqual(self.metrics.dot_product(self.vec_a, self.vec_c), 14)

    def test_euclidean_distance(self):
        """Tests the Euclidean distance calculation."""
        self.assertAlmostEqual(self.metrics.euclidean_distance(self.vec_a, self.vec_b), 5.1961524)
        self.assertEqual(self.metrics.euclidean_distance(self.vec_a, self.vec_c), 0)

    def test_manhattan_distance(self):
        """Tests the Manhattan distance calculation."""
        # The function returns a similarity score, not the raw distance
        distance = np.sum(np.abs(self.vec_a - self.vec_b)) # 3 + 3 + 3 = 9
        expected_similarity = 1 / (1 + distance)
        self.assertAlmostEqual(self.metrics.manhattan_distance(self.vec_a, self.vec_b), expected_similarity)
        # Test identical vectors (distance = 0, similarity = 1)
        self.assertEqual(self.metrics.manhattan_distance(self.vec_a, self.vec_c), 1.0)

    def test_justify_metric_selection(self):
        """Ensures the justification method runs and returns a dictionary."""
        justification = self.metrics.justify_metric_selection()
        self.assertIsInstance(justification, dict)
        self.assertIn('selected_metric', justification)
        self.assertEqual(justification['selected_metric'], 'cosine_similarity')

    def test_get_metric_characteristics(self):
        """Ensures the characteristics method runs and returns a dictionary."""
        characteristics = self.metrics.get_metric_characteristics()
        self.assertIsInstance(characteristics, dict)
        self.assertIn('cosine_similarity', characteristics)
        self.assertIn('range', characteristics['cosine_similarity'])

if __name__ == '__main__':
    unittest.main()
