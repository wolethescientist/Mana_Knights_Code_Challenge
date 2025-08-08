# pipelines/ocr_inference_pipeline.py

import os
import sys
from typing import Any, Dict

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.ocr.ocr_service import OCRService
from services.recommendation_service.product_recommendation_service import ProductRecommendationService


class OcrInferencePipeline:
    """Pipeline for processing OCR queries and getting recommendations."""

    def __init__(self, recommendation_service: ProductRecommendationService):
        """Initializes the pipeline with required services."""
        self.ocr_service = OCRService()
        self.recommendation_service = recommendation_service

    def _clean_extracted_text(self, text: str) -> str:
        """Cleans the raw text extracted by the OCR service."""
        if not text:
            return ""
        cleaned = ' '.join(text.split())
        artifacts = ['|', '~', '`', '^']
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, ' ')
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()

    def run(self, image_data: Any) -> Dict[str, Any]:
        """Executes the full OCR-to-recommendation pipeline."""
        try:
            extracted_text, confidence = self.ocr_service.extract_text_from_image(image_data)

            if not extracted_text:
                return {
                    "error": "No text found in image",
                    "response": "I couldn't extract any text from the image."
                }

            if not self.ocr_service.is_text_readable(extracted_text):
                return {
                    "error": "Text not readable as product query",
                    "response": f"I extracted '{extracted_text}', but it doesn't appear to be a valid query."
                }

            processed_query = self._clean_extracted_text(extracted_text)

            if not self.recommendation_service:
                return {
                    "error": "Recommendation service not available",
                    "response": "The recommendation service is currently unavailable."
                }

            products, response = self.recommendation_service.get_recommendations(processed_query)
            
            enhanced_response = f"From your query '{processed_query}', {response}"

            return {
                "extracted_text": extracted_text,
                "ocr_confidence": confidence,
                "products": products,
                "response": enhanced_response,
                "query": processed_query,
                "total_found": len(products)
            }

        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}",
                "response": "An unexpected error occurred while processing your request."
            }
