"""
Service for product detection using a pre-trained CNN model.

This module defines the `CNNProductDetectionService` class, which is responsible
for loading a trained Keras model and a label encoder, preprocessing images,
and predicting product categories from those images.
"""
import os
from typing import Any, Dict

from pipelines.cnn_inference_pipeline import CNNInferencePipeline


class CNNProductDetectionService:
    """Service for product detection that uses the CNNInferencePipeline."""

    def __init__(self):
        """Initializes the service by creating an instance of the pipeline."""
        try:
            self.pipeline = CNNInferencePipeline()
            self._is_ready = True
            print("CNN Inference Pipeline loaded successfully.")
        except Exception as e:
            self.pipeline = None
            self._is_ready = False
            print(f"Error initializing CNN Inference Pipeline: {e}")

    def predict_product(self, image_file: Any) -> Dict[str, Any]:
        """Predicts the product category by running the inference pipeline."""
        if not self.is_model_ready():
            raise RuntimeError("CNNInferencePipeline is not available.")
        
        return self.pipeline.run(image_file)

    def is_model_ready(self) -> bool:
        """Checks if the inference pipeline was initialized successfully."""
        return self._is_ready
