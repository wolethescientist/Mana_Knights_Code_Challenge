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
import time
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, Response
from werkzeug.datastructures import FileStorage

from services.image_detection_service.product_detection_service import CNNProductDetectionService
from services.ocr.ocr_query_service import OCRQueryService
from services.recommendation_service.product_recommendation_service import ProductRecommendationService
from utils.logging_config import setup_logging
from pipelines.recommendation_data_pipeline import RecommendationDataPipeline

# Load environment variables from a .env file
load_dotenv()

# Set up logging
setup_logging()

app = Flask(__name__, template_folder='frontend')

# Configure Flask for file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

# Initialize services
recommendation_service: Optional[ProductRecommendationService] = None
ocr_query_service: Optional[OCRQueryService] = None
cnn_service: Optional[CNNProductDetectionService] = None

# Get specific loggers
performance_logger = logging.getLogger('performance')

try:
    logging.info("Initializing Product Recommendation Service...")
    start_time = time.time()
    recommendation_service = ProductRecommendationService()
    end_time = time.time()
    performance_logger.info(
        "ProductRecommendationService initialized",
        extra={'extra_data': {'duration_ms': (end_time - start_time) * 1000}}
    )
    logging.info("Product Recommendation Service initialized successfully.")

    # Ensure the vector store is populated. If empty, run the data pipeline once.
    try:
        stats = recommendation_service.vector_store.get_index_stats()
        total_vectors = stats.get('total_vector_count', 0) if isinstance(stats, dict) else 0
        if not total_vectors or total_vectors == 0:
            logging.info("Vector index is empty. Running RecommendationDataPipeline to populate vectors...")
            RecommendationDataPipeline().run()
            logging.info("RecommendationDataPipeline completed and vectors populated.")
        else:
            logging.info(f"Vector index already populated with {total_vectors} vectors. Skipping data pipeline.")
    except Exception as e:
        logging.warning(f"Could not verify/populate vector index. Proceeding without pipeline run. Details: {e}")

    logging.info("Initializing OCR Query Service...")
    start_time = time.time()
    ocr_query_service = OCRQueryService(recommendation_service)
    end_time = time.time()
    performance_logger.info(
        "OCRQueryService initialized",
        extra={'extra_data': {'duration_ms': (end_time - start_time) * 1000}}
    )
    logging.info("OCR Query Service initialized successfully.")

    logging.info("Initializing CNN Product Detection Service...")
    start_time = time.time()
    cnn_service = CNNProductDetectionService()
    end_time = time.time()
    performance_logger.info(
        "CNNProductDetectionService initialized",
        extra={'extra_data': {'duration_ms': (end_time - start_time) * 1000}}
    )
    if cnn_service.is_model_ready():
        logging.info("CNN Product Detection Service initialized successfully.")
    else:
        logging.warning("CNN model or label encoder not found. Detection service will not be available.")

except ImportError as ie:
    logging.error(f"Error: A required library is missing. Please run 'pip install -r requirements.txt'. Details: {ie}")
except Exception as e:
    logging.critical(f"Fatal Error: Could not initialize one or more services. See details below:\n{e}")


def is_valid_image_format(filename: str) -> bool:
    """Checks if the file has a valid image extension."""
    if not filename:
        return False
    return any(filename.lower().endswith(ext) for ext in app.config['UPLOAD_EXTENSIONS'])


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
    api_logger = logging.getLogger('api')
    query = request.form.get('query', '')
    api_logger.info(
        "Product recommendation request received", 
        extra={'extra_data': {'endpoint': request.path, 'method': request.method, 'query': query}}
    )
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not recommendation_service:
        return jsonify({"error": "Recommendation service not available"}), 500
    
    try:
        products, response = recommendation_service.get_recommendations(query)
        
        response_data = {
            "products": products,
            "response": response,
            "query": query,
            "total_found": len(products)
        }
        api_logger.info(
            "Product recommendation response sent",
            extra={'extra_data': {'status_code': 200, 'products_found': len(products)}}
        )
        return jsonify(response_data)
        
    except Exception as e:
        logging.error(f"Error during product recommendation: {e}")
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
    api_logger = logging.getLogger('api')
    
    if 'image_data' not in request.files:
        api_logger.warning("OCR query failed: No image file provided", extra={'extra_data': {'endpoint': request.path, 'method': request.method}})
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image_data']
    
    api_logger.info(
        "OCR query request received",
        extra={'extra_data': {'endpoint': request.path, 'method': request.method, 'filename': image_file.filename}}
    )

    if not image_file.filename or not is_valid_image_format(image_file.filename):
        return jsonify({"error": "Invalid or missing image file"}), 400

    if not ocr_query_service:
        return jsonify({"error": "OCR service is not available"}), 503

    try:
        result = ocr_query_service.process_image_query(image_file)
        if 'error' in result:
            status_code = 400 if result.get('total_found') == 0 else 500
            return jsonify(result), status_code

        api_logger.info(
            "OCR query response sent",
            extra={'extra_data': {'status_code': 200, 'products_found': result.get('total_found')}}
        )
        return jsonify(result)

    except Exception as e:
        logging.error(f"An unexpected error occurred in ocr_query: {e}")
        return jsonify({
            "error": "An unexpected error occurred while processing your request.",
            "details": str(e)
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
    api_logger = logging.getLogger('api')
    
    if 'product_image' not in request.files:
        api_logger.warning("Image search failed: No product image provided", extra={'extra_data': {'endpoint': request.path, 'method': request.method}})
        return jsonify({"error": "No product image provided"}), 400

    product_image = request.files['product_image']
    
    api_logger.info(
        "Image product search request received",
        extra={'extra_data': {'endpoint': request.path, 'method': request.method, 'filename': product_image.filename}}
    )

    if not product_image.filename or not is_valid_image_format(product_image.filename):
        return jsonify({"error": "Invalid or missing image file"}), 400

    if not cnn_service or not cnn_service.is_model_ready():
        return jsonify({
            "error": "CNN service not available",
            "response": "The image detection service is currently unavailable."
        }), 503

    try:
        prediction_result = cnn_service.predict_product(product_image)
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result['confidence']
        top_3_predictions = prediction_result['top_3_predictions']

        products = []
        response = f"I detected a '{predicted_class}' in your image with {confidence:.2%} confidence."

        if recommendation_service:
            try:
                matching_products, _ = recommendation_service.get_recommendations(predicted_class)
                products = matching_products
                if products:
                    response += f" I found {len(products)} matching products for you."
                else:
                    response += " However, I couldn't find any matching products in our database."
            except Exception as e:
                logging.error(f"Error getting product recommendations: {e}")
                response += " However, I encountered an issue while searching for matching products."

        response_data = {
            "products": products,
            "response": response,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_3_predictions": top_3_predictions
        }
        api_logger.info(
            "Image product search response sent",
            extra={'extra_data': {'status_code': 200, 'predicted_class': predicted_class}}
        )
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in image product search: {e}")
        return jsonify({
            "error": "An error occurred while processing your image",
            "response": "Sorry, I encountered an error while analyzing your image. Please try again."
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
    return render_template('search_text.html')

@app.route('/ocr-query', methods=['GET'])
def ocr_query_page() -> str:
    """
    Renders the page for OCR-based query processing.

    This page allows users to upload an image of a handwritten note to be used
    as a search query.
    """
    return render_template('search_ocr.html')

@app.route('/image-detection', methods=['GET'])
def image_detection_page() -> str:
    """
    Renders the page for image-based product detection.

    This page allows users to upload a product image, which the CNN model will
    analyze to identify the product category.
    """
    return render_template('search_image.html')

@app.route('/services', methods=['GET'])
def services_page() -> str:
    """
    Renders the e-commerce services page.

    This route serves the 'services.html' template.
    """
    return render_template('services.html')

if __name__ == '__main__':
    # This block runs the Flask development server only when the script is executed directly.
    # The debug=True flag enables auto-reloading and an interactive debugger.
    logging.info("--- Application starting ---")
    logging.info("URL: http://127.0.0.1:5000")
    logging.info("Dashboard: http://127.0.0.1:5000/")
    logging.info("--------------------------")
    app.run(debug=True)