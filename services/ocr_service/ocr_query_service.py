"""
Service that orchestrates OCR text extraction and product recommendation.

This module defines the `OCRQueryService`, which uses the `OCRService` to extract
text from an image and then passes the cleaned text to the
`ProductRecommendationService` to find relevant products.
"""
import logging
from typing import Any, Dict, Optional

from ..recommendation_service.product_recommendation_service import ProductRecommendationService
from .ocr_service import OCRService


class OCRQueryService:
    """Orchestrates OCR processing and product recommendation.

    This service acts as a high-level interface to process an image containing
    a handwritten or printed query. It extracts the text, cleans it, and then
    queries the recommendation service to find matching products.

    Attributes:
        logger: A logger for recording events and errors.
        ocr_service (OCRService): An instance of the core OCR service.
        recommendation_service (ProductRecommendationService): An instance of the
            product recommendation service.
    """

    def __init__(self, recommendation_service: Optional[ProductRecommendationService] = None) -> None:
        """Initializes the OCRQueryService.

        Args:
            recommendation_service: An optional instance of the
                ProductRecommendationService. If not provided, a new instance
                will be created.
        """
        self.logger = logging.getLogger(__name__)
        self.ocr_service = OCRService()
        self.recommendation_service = recommendation_service
        
        if not self.recommendation_service:
            try:
                self.recommendation_service = ProductRecommendationService()
                self.logger.info("Product recommendation service initialized successfully")
            except Exception as e:
                self.logger.error(f"Could not initialize recommendation service: {e}")
                self.recommendation_service = None

    def process_image_query(self, image_data: Any) -> Dict[str, Any]:
        """Processes an image query to extract text and find products.

        This method takes image data, extracts text using the OCR service, cleans
        the text, and then queries the recommendation service for products.

        Args:
            image_data: A file-like object or binary data of the image.

        Returns:
            A dictionary containing the results of the operation, including
            extracted text, products found, and a response message. In case of
            an error, it includes an 'error' key.
        """
        try:
            # Extract text from image
            extracted_text, confidence = self.ocr_service.extract_text_from_image(image_data)
            
            # Log extraction results for debugging
            self.logger.info(f"OCR extracted: '{extracted_text}' with confidence: {confidence:.1f}%")

            # Check if text extraction was successful
            if not extracted_text:
                return {
                    "extracted_text": extracted_text,
                    "ocr_confidence": confidence,
                    "products": [],
                    "response": "I couldn't extract any text from the image. Please ensure the image contains clear, legible text and try again.",
                    "query": "",
                    "total_found": 0,
                    "error": "No text found in image"
                }

            # Check if text is readable
            if not self.ocr_service.is_text_readable(extracted_text):
                return {
                    "extracted_text": extracted_text,
                    "ocr_confidence": confidence,
                    "products": [],
                    "response": f"I extracted '{extracted_text}' from the image, but it doesn't appear to be a valid product query. Please try with a clearer image.",
                    "query": "",
                    "total_found": 0,
                    "error": "Text not readable as product query"
                }
            
            # Clean and process the extracted text
            processed_query = self._clean_extracted_text(extracted_text)

            # If no processed query, try using the raw extracted text as fallback
            if not processed_query:
                processed_query = extracted_text.strip()

            if not processed_query:
                return {
                    "extracted_text": extracted_text,
                    "ocr_confidence": confidence,
                    "products": [],
                    "response": "The extracted text doesn't appear to be a valid product query. Please try with a clearer image containing a product search query.",
                    "query": processed_query,
                    "total_found": 0,
                    "error": "Invalid query text"
                }
            
            # Get product recommendations using the extracted text
            if self.recommendation_service:
                try:
                    products, response = self.recommendation_service.get_recommendations(processed_query)
                    
                    # Enhance the response to mention OCR extraction
                    enhanced_response = f"From your handwritten query '{processed_query}', {response}"
                    
                    return {
                        "extracted_text": extracted_text,
                        "ocr_confidence": confidence,
                        "products": products,
                        "response": enhanced_response,
                        "query": processed_query,
                        "total_found": len(products)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error getting recommendations: {e}")
                    return {
                        "extracted_text": extracted_text,
                        "ocr_confidence": confidence,
                        "products": [],
                        "response": f"I extracted '{processed_query}' from your image, but encountered an error while searching for products. Please try again.",
                        "query": processed_query,
                        "total_found": 0,
                        "error": "Recommendation service error"
                    }
            else:
                return {
                    "extracted_text": extracted_text,
                    "ocr_confidence": confidence,
                    "products": [],
                    "response": f"I extracted '{processed_query}' from your image, but the product recommendation service is not available.",
                    "query": processed_query,
                    "total_found": 0,
                    "error": "Recommendation service not available"
                }
                
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

    def _clean_extracted_text(self, text: str) -> str:
        """Cleans the raw text extracted by the OCR service.

        This method removes common OCR artifacts and extra whitespace to prepare
        the text for use as a search query.

        Args:
            text: The raw string extracted from the image.

        Returns:
            A cleaned string suitable for a product search query.
        """
        if not text:
            return ""

        # Remove extra whitespace and normalize
        cleaned = ' '.join(text.split())

        # Remove common OCR artifacts but be more conservative
        artifacts = ['|', '~', '`', '^']
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, ' ')

        # Remove multiple spaces
        cleaned = ' '.join(cleaned.split())

        # Basic validation - ensure it's not empty
        if len(cleaned.strip()) < 1:
            return ""

        # Allow single words and numbers (they could be valid product queries)
        return cleaned.strip()

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
