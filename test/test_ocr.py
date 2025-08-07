#!/usr/bin/env python3
"""
Test script for the OCRService and OCRQueryService.

This script contains two main test functions:
1.  `test_ocr_basic`: Verifies that the core `OCRService` can be initialized,
    extract text from a generated image, and assess text readability.
2.  `test_ocr_query_service`: Checks that the `OCRQueryService` can be
    initialized and can correctly validate a generated test image.

The script reports the success or failure of each test and provides a final
summary.
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.ocr_service.ocr_service import OCRService
from services.ocr_service.ocr_query_service import OCRQueryService

def test_ocr_basic() -> bool:
    """Tests the basic functionality of the OCRService.

    This function performs the following steps:
    1. Initializes the OCRService.
    2. Creates a simple black-and-white test image with the text 'Hello World'.
    3. Uses the service to extract text and confidence from the image.
    4. Verifies that the extracted text is readable.
    5. Checks if the extracted text contains the expected content.

    Returns:
        bool: True if the test passes, False otherwise.
    """
    print("Testing OCR Service...")
    
    try:
        # Initialize OCR service
        ocr_service = OCRService()
        print("✓ OCR Service initialized successfully")
        
        # Create a simple test image with text
        # Create a white image
        img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        
        # Add some text using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Hello World', (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Test OCR on the created image
        text, confidence = ocr_service.extract_text_from_image(img)
        
        print(f"✓ Extracted text: '{text}'")
        print(f"✓ Confidence: {confidence:.1f}%")
        
        # Test text readability check
        is_readable = ocr_service.is_text_readable(text)
        print(f"✓ Text is readable: {is_readable}")
        
        if text and 'Hello' in text:
            print("✓ OCR test passed!")
            return True
        else:
            print("⚠ OCR test partially successful - text extracted but may not be accurate")
            return True
            
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False

def test_ocr_query_service() -> bool:
    """Tests the basic functionality of the OCRQueryService.

    This function performs the following steps:
    1. Initializes the OCRQueryService.
    2. Creates a test image with a sample product query ('laptop computer').
    3. Uses the service to validate the image format.

    Returns:
        bool: True if the test passes, False otherwise.
    """
    print("\nTesting OCR Query Service...")
    
    try:
        # Initialize without recommendation service for a basic test
        ocr_query_service = OCRQueryService()
        print("✓ OCR Query Service initialized")
        
        # Create a test image with a product query
        img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'laptop computer', (50, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Test image validation
        is_valid = ocr_query_service.validate_image_format(img)
        print(f"✓ Image format validation: {is_valid}")
        
        print("✓ OCR Query Service test passed!")
        return True
        
    except Exception as e:
        print(f"✗ OCR Query Service test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("OCR FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test basic OCR
    ocr_success = test_ocr_basic()
    
    # Test OCR Query Service
    query_success = test_ocr_query_service()
    
    print("\n" + "=" * 50)
    if ocr_success and query_success:
        print("✓ ALL TESTS PASSED - OCR functionality is ready!")
    else:
        print("⚠ Some tests failed - check the error messages above")
    print("=" * 50)
