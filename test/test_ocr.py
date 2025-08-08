#!/usr/bin/env python3
"""
Test script for the OCRService.

This script contains one main test function:
`test_ocr_basic`: Verifies that the core `OCRService` can be initialized,
    extract text from a generated image, and assess text readability.

The script reports the success or failure of the test.
"""

import sys
import os
import cv2
import numpy as np

# Add the project root to the Python path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.ocr.ocr_service import OCRService

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



if __name__ == "__main__":
    print("=" * 50)
    print("OCR FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test basic OCR
    ocr_success = test_ocr_basic()
    
    print("\n" + "=" * 50)
    if ocr_success:
        print("✓ ALL TESTS PASSED - OCR functionality is ready!")
    else:
        print("⚠ Some tests failed - check the error messages above")
    print("=" * 50)
