"""
Core OCR service for text extraction from images.

This module provides the `OCRService` class, which encapsulates the logic for
preprocessing images and extracting text using the Tesseract OCR engine. It is
optimized for handling handwritten text by trying multiple preprocessing
strategies and selecting the best result.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image


class OCRService:
    """A service for extracting text from images using Tesseract OCR.

    This class provides a comprehensive suite of tools for OCR, including
    advanced image preprocessing, multiple extraction strategies, and intelligent
    result scoring to maximize accuracy, especially for handwritten text.

    Attributes:
        logger: A logger for recording operational information and errors.
    """

    def __init__(self) -> None:
        """Initializes the OCRService.

        This constructor sets up the logger and configures the path to the
        Tesseract executable by searching common installation directories.
        """
        self.logger = logging.getLogger(__name__)

        # Set Tesseract path - try common Windows installation paths
        possible_paths = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            "tesseract"  # If it's in PATH
        ]

        tesseract_path = None
        for path in possible_paths:
            if path == "tesseract" or os.path.exists(path):
                tesseract_path = path
                break

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.logger.info(f"Tesseract path set to: {tesseract_path}")
        else:
            self.logger.warning("Tesseract not found in common paths. Make sure it's installed and in PATH.")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses an image to improve OCR accuracy.

        This method applies a series of image processing techniques, including
        grayscale conversion, contrast enhancement (CLAHE), Gaussian blur,
        adaptive thresholding, denoising, and deskewing.

        Args:
            image: A NumPy array representing the input image.

        Returns:
            A NumPy array representing the preprocessed image.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Apply Gaussian blur to reduce noise before thresholding
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

            # Try multiple thresholding approaches and pick the best one
            # Method 1: Adaptive thresholding with Gaussian
            thresh1 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Method 2: Adaptive thresholding with Mean
            thresh2 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Method 3: Otsu's thresholding
            _, thresh3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Choose the threshold method that produces the most balanced result
            # (not too much black or white pixels)
            def evaluate_threshold(thresh_img):
                white_pixels = np.sum(thresh_img == 255)
                total_pixels = thresh_img.size
                white_ratio = white_pixels / total_pixels
                # Prefer images with 70-90% white pixels (text on white background)
                if 0.7 <= white_ratio <= 0.9:
                    return abs(0.8 - white_ratio)  # Closer to 80% is better
                else:
                    return 1.0  # Penalize heavily

            scores = [evaluate_threshold(t) for t in [thresh1, thresh2, thresh3]]
            best_thresh = [thresh1, thresh2, thresh3][np.argmin(scores)]

            # Light denoising (less aggressive for handwritten text)
            denoised = cv2.medianBlur(best_thresh, 3)

            # Deskew the image by calculating the rotation angle using minAreaRect
            try:
                coords = np.column_stack(np.where(denoised == 0))  # Find black pixels (text)
                if len(coords) > 10:  # Need sufficient points for reliable angle calculation
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle

                    # Only apply rotation if angle is significant (> 1 degree)
                    if abs(angle) > 1.0:
                        (h, w) = denoised.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        deskewed_img = cv2.warpAffine(denoised, M, (w, h),
                                                    flags=cv2.INTER_CUBIC,
                                                    borderMode=cv2.BORDER_REPLICATE)
                        denoised = deskewed_img
            except Exception as e:
                self.logger.debug(f"Deskewing failed, using original: {e}")

            # Very light morphological operations (preserve handwritten character details)
            kernel = np.ones((1, 1), np.uint8)  # Smaller kernel for handwritten text
            processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

            # Resize image to improve OCR accuracy (larger scale factor for handwritten text)
            height, width = processed.shape
            scale_factor = 3  # Increased scale factor for better handwritten text recognition
            resized = cv2.resize(
                processed, (width * scale_factor, height * scale_factor),
                interpolation=cv2.INTER_CUBIC
            )

            return resized

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return image

    def extract_text_from_image(self, image_data: Any, preprocess: bool = True) -> Tuple[str, float]:
        """
        Extract text from image using OCR optimized for handwritten text.
        Uses multiple preprocessing approaches and cropping strategies for better accuracy.

        Args:
            image_data: Image data (file path, PIL Image, numpy array, or file-like object)
            preprocess: Whether to preprocess the image for better OCR

        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Step 1: Load image properly
            image = self._load_image(image_data)
            if image is None:
                raise ValueError("Could not load or decode image data")

            # Convert to PIL Image for cropping operations
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Step 2: Try multiple extraction strategies and pick the best result
            results = []

            # Strategy 1: Multiple preprocessing approaches
            preprocessing_results = self._try_multiple_preprocessing(pil_image, preprocess)
            results.extend(preprocessing_results)

            # Strategy 2: Try different crops to handle cut-off text
            cropping_results = self._try_image_cropping(pil_image, preprocess)
            results.extend(cropping_results)

            # Step 3: Score and select the best result
            best_text, best_confidence = self._select_best_result(results)

            self.logger.info(f"Final extracted text: '{best_text}' with confidence: {best_confidence:.1f}%")
            return best_text, best_confidence

        except Exception as e:
            self.logger.error(f"Error extracting text from image: {e}")
            return "", 0.0

    def _load_image(self, image_data: Any) -> Optional[Image.Image]:
        """Load image from various input types."""
        try:
            if isinstance(image_data, str):
                # File path
                return cv2.imread(image_data)
            elif isinstance(image_data, Image.Image):
                # PIL Image
                return cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            elif isinstance(image_data, np.ndarray):
                # Numpy array
                return image_data
            else:
                # Handle file-like objects (Flask FileStorage)
                if hasattr(image_data, 'read'):
                    file_bytes = image_data.read()
                    if hasattr(image_data, 'seek'):
                        image_data.seek(0)
                else:
                    file_bytes = image_data

                # Convert bytes to numpy array and decode
                nparr = np.frombuffer(file_bytes, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None

    def _simple_preprocess(self, gray_image: np.ndarray) -> Image.Image:
        """Simple preprocessing optimized for handwritten text."""
        try:
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray_image)

            # Light blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Scale up for better OCR
            height, width = thresh.shape
            scaled = cv2.resize(thresh, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

            return scaled
        except Exception as e:
            self.logger.error(f"Error in simple preprocessing: {e}")
            return gray_image

    def _light_preprocess(self, pil_image: Image.Image) -> Image.Image:
        """Light preprocessing - minimal changes to preserve text integrity."""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Simple contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Convert back to PIL Image
            return Image.fromarray(enhanced)
        except Exception as e:
            self.logger.error(f"Error in light preprocessing: {e}")
            return pil_image

    def _aggressive_preprocess(self, pil_image: Image.Image) -> Image.Image:
        """Aggressive preprocessing with adaptive thresholding, deskewing, and morphological operations."""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            # Deskewing using minAreaRect angle calculation
            coords = np.column_stack(np.where(binary == 0))  # Find black pixels (text)
            if len(coords) > 10:
                rect = cv2.minAreaRect(coords)
                angle = rect[2]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle

                # Rotate image to correct skew
                if abs(angle) > 0.5:  # Only rotate if angle is significant
                    (h, w) = binary.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_REPLICATE)

            # Morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Convert back to PIL Image
            return Image.fromarray(binary)
        except Exception as e:
            self.logger.error(f"Error in aggressive preprocessing: {e}")
            return pil_image



    def is_text_readable(self, text: str, min_length: int = 1) -> bool:
        """Checks if the extracted text is likely to be a meaningful query.

        This method uses heuristics to determine if the text is readable, such as
        checking its length, character variety, and the ratio of alphanumeric
        characters.

        Args:
            text: The text extracted from the OCR process.
            min_length: The minimum number of characters for the text to be
                considered readable.

        Returns:
            True if the text is considered readable, False otherwise.
        """
        if not text or len(text.strip()) < min_length:
            return False

        # Clean the text
        cleaned_text = text.strip()

        # Very basic check - just needs to have at least one letter
        has_letter = any(c.isalpha() for c in cleaned_text)

        # Check if it's not just random symbols
        alphanumeric_count = sum(c.isalnum() for c in cleaned_text)
        total_chars = len(cleaned_text.replace(' ', ''))

        if total_chars == 0:
            return False

        # More lenient - at least 30% alphanumeric OR has letters (for handwritten text)
        alphanumeric_ratio = (alphanumeric_count / total_chars) if total_chars > 0 else 0

        # Accept if it has letters and at least some alphanumeric content
        return has_letter and (alphanumeric_ratio >= 0.3 or len(cleaned_text) >= 2)

    def _try_multiple_preprocessing(self, pil_image: Image.Image, preprocess: bool) -> List[Tuple[str, str, float]]:
        """Try multiple preprocessing approaches and return results."""
        results = []

        try:
            # Approach 1: Original image (no preprocessing)
            self.logger.debug("Trying original image...")
            text, conf = self._extract_with_tesseract(pil_image, "original")
            if text:
                results.append(("original", text, conf))

            if preprocess:
                # Approach 2: Light preprocessing
                self.logger.debug("Trying light preprocessing...")
                light_processed = self._light_preprocess(pil_image)
                text, conf = self._extract_with_tesseract(light_processed, "light")
                if text:
                    results.append(("light", text, conf))

                # Approach 3: Aggressive preprocessing
                self.logger.debug("Trying aggressive preprocessing...")
                aggressive_processed = self._aggressive_preprocess(pil_image)
                text, conf = self._extract_with_tesseract(aggressive_processed, "aggressive")
                if text:
                    results.append(("aggressive", text, conf))

        except Exception as e:
            self.logger.error(f"Error in multiple preprocessing: {e}")

        return results

    def _try_image_cropping(self, pil_image: Image.Image, preprocess: bool) -> List[Tuple[str, str, float]]:
        """Try different crops of the image to handle cut-off text."""
        results = []

        try:
            width, height = pil_image.size

            # Define different crops to try
            crops = [
                # Focus on left portion (in case right side is cut off)
                (0, 0, int(width * 0.8), height, "left_crop"),
                # Focus on right portion (in case left side is cut off)
                (int(width * 0.2), 0, width, height, "right_crop"),
                # Focus on center portion
                (int(width * 0.1), 0, int(width * 0.9), height, "center_crop"),
                # Slightly expand bounds (if possible)
                (max(0, -5), max(0, -5), min(width + 5, width), min(height + 5, height), "expanded_crop")
            ]

            for left, top, right, bottom, crop_name in crops:
                try:
                    # Ensure crop bounds are valid
                    left = max(0, left)
                    top = max(0, top)
                    right = min(width, right)
                    bottom = min(height, bottom)

                    if right > left and bottom > top:
                        cropped = pil_image.crop((left, top, right, bottom))

                        # Try both light and aggressive preprocessing on crops
                        if preprocess:
                            # Light preprocessing on crop
                            light_crop = self._light_preprocess(cropped)
                            text, conf = self._extract_with_tesseract(light_crop, f"{crop_name}_light")
                            if text:
                                results.append((f"{crop_name}_light", text, conf))
                        else:
                            # Original crop
                            text, conf = self._extract_with_tesseract(cropped, crop_name)
                            if text:
                                results.append((crop_name, text, conf))

                except Exception as e:
                    self.logger.debug(f"Crop {crop_name} failed: {e}")

        except Exception as e:
            self.logger.error(f"Error in image cropping: {e}")

        return results

    def _extract_with_tesseract(self, pil_image: Image.Image, method_name: str) -> Tuple[str, float]:
        """Extract text using Tesseract with multiple PSM modes."""
        try:
            # Try default Tesseract first
            text = pytesseract.image_to_string(pil_image).strip()
            confidence = 50.0

            if text:
                # Calculate confidence
                try:
                    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    confidence = sum(confidences) / len(confidences) if confidences else 50.0
                except:
                    confidence = 50.0

                self.logger.debug(f"{method_name} OCR successful: '{text}' (confidence: {confidence:.1f}%)")
                return text, confidence
            else:
                # Try alternative PSM modes for handwritten text
                psm_configs = [
                    ('--psm 8', 40.0),  # Single word
                    ('--psm 13', 30.0), # Raw line
                    ('--psm 6', 35.0),  # Single uniform block
                ]

                for config, conf_score in psm_configs:
                    try:
                        alt_text = pytesseract.image_to_string(pil_image, config=config).strip()
                        if alt_text:
                            self.logger.debug(f"{method_name} {config} OCR successful: '{alt_text}'")
                            return alt_text, conf_score
                    except:
                        continue

                return "", 0.0

        except Exception as e:
            self.logger.error(f"Tesseract extraction failed for {method_name}: {e}")
            return "", 0.0

    def _select_best_result(self, results: List[Tuple[str, str, float]]) -> Tuple[str, float]:
        """Select the best result from multiple OCR attempts using intelligent scoring."""
        if not results:
            return "", 0.0

        # Score each result
        scored_results = []
        for method, text, confidence in results:
            if not text.strip():
                continue

            score = 0

            # Base score from confidence
            score += confidence * 0.5

            # Length bonus (longer text often means more complete extraction)
            word_count = len(text.split())
            score += word_count * 3

            # Character count bonus
            char_count = len(text.strip())
            score += char_count * 0.5

            # Bonus for proper sentence structure
            if text and text[0].isupper() and word_count > 1:
                score += 10

            # Bonus for containing common product-related words
            product_words = ['laptop', 'computer', 'phone', 'tablet', 'smartphone', 'antique', 'suggest', 'some']
            for word in product_words:
                if word.lower() in text.lower():
                    score += 15

            # Penalty for obvious OCR errors (too many special characters)
            special_char_count = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-')
            if special_char_count > len(text) * 0.3:
                score -= 20

            # Penalty for too many digits (unless it's a product code)
            digit_count = sum(1 for c in text if c.isdigit())
            if digit_count > len(text) * 0.5 and word_count < 2:
                score -= 15

            scored_results.append((score, method, text, confidence))
            self.logger.debug(f"Scored {method}: {score:.1f} points for '{text}'")

        if scored_results:
            # Sort by score (highest first)
            scored_results.sort(reverse=True)
            best_score, best_method, best_text, best_confidence = scored_results[0]

            self.logger.info(f"Best result from {best_method}: '{best_text}' (score: {best_score:.1f}, confidence: {best_confidence:.1f}%)")
            return best_text, best_confidence
        else:
            return "", 0.0


# Standalone functions for direct usage (keeping your preferred format)
def preprocess_image(image_path: str) -> np.ndarray:
    """Standalone preprocessing function using the improved OCR service."""
    ocr_service = OCRService()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    return ocr_service.preprocess_image(img)

def extract_text(image_path: str) -> str:
    """Standalone text extraction function using the improved OCR service."""
    try:
        ocr_service = OCRService()
        text, confidence = ocr_service.extract_text_from_image(image_path)
        return text if text else f"No text found (confidence: {confidence:.1f}%)"
    except Exception as e:
        return f"Error processing image: {str(e)}"


