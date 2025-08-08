"""
Service that orchestrates OCR text extraction and product recommendation.

This module defines the `OCRQueryService`, which uses the `OcrInferencePipeline` to extract
text from an image and then passes the cleaned text to the
`ProductRecommendationService` to find relevant products.
"""
import logging
from typing import Any, Dict, Optional

from pipelines.ocr_inference_pipeline import OcrInferencePipeline
from services.recommendation_service.product_recommendation_service import ProductRecommendationService


class OCRQueryService:
    """Service for processing OCR queries using the OcrInferencePipeline."""

    def __init__(self, recommendation_service: ProductRecommendationService):
        """Initializes the service with an instance of the OCR pipeline."""
        self.logger = logging.getLogger(__name__)
        self.pipeline = OcrInferencePipeline(recommendation_service)

    def process_image_query(self, image_data: Any) -> Dict[str, Any]:
        """Processes an image query by running the OCR inference pipeline."""
        try:
            return self.pipeline.run(image_data)
        except Exception as e:
            self.logger.error(f"Error processing image query: {e}")
            return {
                "extracted_text": "",
                "ocr_confidence": 0.0,
                "products": [],
                "response": "An error occurred while processing your image. Please ensure the image is valid and try again.",
                "query": "",
                "total_found": 0,
                "error": f"Processing error: {str(e)}"
            }

    def validate_image_format(self, image_data: Any) -> bool:
        """Validates if the uploaded image has a supported format.

        Checks the file extension of the uploaded image against a list of
        supported formats.

        Args:
            image_data: A file-like object from a file upload.

        Returns:
            True if the image format is supported, False otherwise.
        """
        try:
            # Try to process the image to see if it's valid
            if hasattr(image_data, 'filename'):
                filename = image_data.filename.lower()
                supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
                return any(filename.endswith(ext) for ext in supported_extensions)
            return True  # Assume valid if we can't check filename
        except:
            return False
