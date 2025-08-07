"""
Main Flask application for the E-commerce Services API.

This application provides a web interface and API endpoints for various e-commerce
services, including:
- Product recommendations based on text queries.
- OCR-based queries from handwritten notes.
- Product detection and identification from images using a CNN model.

It initializes and manages the required services and routes requests to the
appropriate handlers. It also serves the HTML pages for user interaction.
"""
import os
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, Response
from werkzeug.datastructures import FileStorage

# Adjust imports to be more explicit and consistent with the project structure
from services.cnn_detection_service.product_detection_service import CNNProductDetectionService
from services.ocr_service.ocr_query_service import OCRQueryService
from services.recommendation_service.product_recommendation_service import ProductRecommendationService
from utils.logging_config import setup_logging

# Load environment variables from a .env file
load_dotenv()

# Set up logging
setup_logging()

app = Flask(__name__)

# Configure Flask for file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

# Initialize services
recommendation_service: Optional[ProductRecommendationService] = None
ocr_query_service: Optional[OCRQueryService] = None
cnn_detection_service: Optional[CNNProductDetectionService] = None

try:
    logging.info("Initializing Product Recommendation Service...")
    recommendation_service = ProductRecommendationService()
    logging.info("Product Recommendation Service initialized successfully.")

    logging.info("Initializing OCR Query Service...")
    ocr_query_service = OCRQueryService(recommendation_service)
    logging.info("OCR Query Service initialized successfully.")

    logging.info("Initializing CNN Product Detection Service...")
    cnn_detection_service = CNNProductDetectionService()
    if cnn_detection_service.is_model_ready():
        logging.info("CNN Product Detection Service initialized successfully.")
    else:
        logging.warning("CNN model or label encoder not found. Detection service will not be available.")

except ImportError as ie:
    logging.error(f"Error: A required library is missing. Please run 'pip install -r requirements.txt'. Details: {ie}")
except Exception as e:
    logging.critical(f"Fatal Error: Could not initialize one or more services. See details below:\n{e}")


@app.route('/product-recommendation', methods=['POST'])
def product_recommendation() -> Response:
    """
    Handles product recommendations based on a natural language query.

    Accepts a POST request with a 'query' in the form data. It uses the
    ProductRecommendationService to find matching products.

    Returns:
        A JSON response containing the list of recommended products, a natural
        language response, the original query, and the total number of products found.
        Returns an error if the service is unavailable or the query is missing.
    """
    query = request.form.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not recommendation_service:
        return jsonify({"error": "Recommendation service not available"}), 500
    
    try:
        # Get recommendations - let the system return whatever number it finds
        products, response = recommendation_service.get_recommendations(query)
        
        return jsonify({
            "products": products,
            "response": response,
            "query": query,
            "total_found": len(products)
        })
        
    except Exception as e:
        return jsonify({
            "error": "An error occurred while processing your request",
            "products": [],
            "response": "Sorry, I encountered an error while searching for products. Please try again."
        }), 500
    
    
@app.route('/ocr-query', methods=['POST'])
def ocr_query() -> Response:
    """
    Processes an uploaded image of a handwritten query using the OCR service.

    Accepts a POST request with an 'image_data' file. The OCR service extracts
    text from the image, which is then used to find product recommendations.

    Returns:
        A JSON response containing recommended products, the extracted text,
        and the OCR confidence score. Returns an error if the service is unavailable
        or no image is provided.
    """
    try:
        # Get the uploaded image file
        image_file = request.files.get('image_data')

        print(f"DEBUG: Received file upload request")
        print(f"DEBUG: Files in request: {list(request.files.keys())}")
        print(f"DEBUG: Image file: {image_file}")

        if not image_file:
            print("DEBUG: No image file provided")
            return jsonify({
                "error": "No image file provided",
                "products": [],
                "response": "Please upload an image file to process.",
                "extracted_text": "",
                "ocr_confidence": 0.0
            }), 400

        # Check file details
        print(f"DEBUG: File name: {image_file.filename}")
        print(f"DEBUG: File content type: {image_file.content_type}")

        # Check if OCR service is available
        if not ocr_query_service:
            print("DEBUG: OCR service not available")
            return jsonify({
                "error": "OCR service not available",
                "products": [],
                "response": "OCR service is currently not available. Please try again later.",
                "extracted_text": "",
                "ocr_confidence": 0.0
            }), 500

        # Validate image format
        if not ocr_query_service.validate_image_format(image_file):
            print(f"DEBUG: Unsupported image format: {image_file.filename}")
            return jsonify({
                "error": "Unsupported image format",
                "products": [],
                "response": "Please upload a supported image format (JPG, PNG, BMP, TIFF, GIF).",
                "extracted_text": "",
                "ocr_confidence": 0.0
            }), 400

        print("DEBUG: Starting image processing...")
        # Process the image and get results
        result = ocr_query_service.process_image_query(image_file)
        print(f"DEBUG: OCR processing completed. Result: {result}")

        # Return the complete result
        return jsonify(result)

    except Exception as e:
        print(f"DEBUG: Exception in ocr_query: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"An error occurred while processing your image: {str(e)}",
            "products": [],
            "response": "Sorry, I encountered an error while processing your image. Please try again.",
            "extracted_text": "",
            "ocr_confidence": 0.0
        }), 500

@app.route('/image-product-search', methods=['POST'])
def image_product_search() -> Response:
    """
    Identifies a product from an uploaded image using the CNN model.

    Accepts a POST request with a 'product_image' file. The CNN service predicts
    the product category, which is then used to query for related products.

    Returns:
        A JSON response containing matching products, the predicted class,
        confidence score, and top-3 predictions. Returns an error if the service
        is unavailable or no image is provided.
    """
    try:
        # Get the uploaded image file
        product_image = request.files.get('product_image')

        if not product_image:
            return jsonify({
                "error": "No image file provided",
                "products": [],
                "response": "Please upload an image file to process.",
                "predicted_class": "",
                "confidence": 0.0
            }), 400

        # Check if CNN service is available
        if not cnn_detection_service or not cnn_detection_service.is_model_ready():
            return jsonify({
                "error": "CNN detection service not available",
                "products": [],
                "response": "CNN model is not available. Please ensure the model is trained and loaded.",
                "predicted_class": "",
                "confidence": 0.0
            }), 500

        # Use CNN model to predict product category
        prediction_result = cnn_detection_service.predict_product(product_image)
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        top_3_predictions = prediction_result['top_3_predictions']

        # Use the predicted class to get matching products from the recommendation service
        products = []
        response = f"I detected a '{predicted_class}' in your image with {confidence:.2%} confidence."

        if recommendation_service:
            try:
                # Search for products matching the predicted class
                matching_products, rec_response = recommendation_service.get_recommendations(predicted_class)
                products = matching_products

                if products:
                    response += f" I found {len(products)} matching products for you."
                else:
                    response += " However, I couldn't find any matching products in our database."
            except Exception as e:
                logging.error(f"Error getting product recommendations: {e}")
                response += " However, I encountered an issue while searching for matching products."

        return jsonify({
            "products": products,
            "response": response,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions
        })

    except Exception as e:
        print(f"Error in image product search: {e}")
        return jsonify({
            "error": "An error occurred while processing your image",
            "products": [],
            "response": "Sorry, I encountered an error while analyzing your image. Please try again.",
            "predicted_class": "",
            "confidence": 0.0
        }), 500




@app.route('/sample_response', methods=['GET'])
def sample_response() -> str:
    """
    Renders a page displaying a sample JSON response from the API.

    This is a utility page for developers to see the format of API responses.
    """
    return render_template('sample_response.html')

@app.route('/', methods=['GET'])
def dashboard() -> str:
    """
    Renders the main application dashboard.

    This page serves as the main entry point and navigation hub for the application.
    """
    return render_template('dashboard.html')

@app.route('/text-search', methods=['GET'])
def text_search_page() -> str:
    """
    Renders the page for text-based product search.

    This page provides a simple interface for users to enter a text query and
    receive product recommendations.
    """
    return render_template('text_search.html')

@app.route('/ocr-query', methods=['GET'])
def ocr_query_page() -> str:
    """
    Renders the page for OCR-based query processing.

    This page allows users to upload an image of a handwritten note to be used
    as a search query.
    """
    return render_template('ocr_query.html')

@app.route('/image-detection', methods=['GET'])
def image_detection_page() -> str:
    """
    Renders the page for image-based product detection.

    This page allows users to upload a product image, which the CNN model will
    analyze to identify the product category.
    """
    return render_template('image_detection.html')

@app.route('/legacy', methods=['GET'])
def ecommerce_services_page() -> str:
    """
    Renders the legacy e-commerce services page.

    This route is maintained for compatibility or historical purposes and serves
    the original 'ecommerce_services.html' template.
    """
    return render_template('ecommerce_services.html')

if __name__ == '__main__':
    # This block runs the Flask development server only when the script is executed directly.
    # The debug=True flag enables auto-reloading and an interactive debugger.
    logging.info("--- Application starting ---")
    logging.info("URL: http://127.0.0.1:5000")
    logging.info("Dashboard: http://127.0.0.1:5000/")
    logging.info("--------------------------")
    app.run(debug=True)
