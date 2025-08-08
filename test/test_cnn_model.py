#!/usr/bin/env python3
"""
Test script for the CNN Product Detection Service.

This script verifies that the CNN model and the associated service can be loaded
correctly and can make predictions on sample images. It loads a few test samples
from the training data, predicts their categories, and compares the predictions
against the true labels.
"""
import os
import sys

import pandas as pd

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.image_detection_service.product_detection_service import CNNProductDetectionService

def test_cnn_model() -> None:
    """Tests the CNN model with a few sample images from the dataset.

    This function performs the following steps:
    1. Verifies that the model and label encoder files exist.
    2. Initializes the CNNProductDetectionService.
    3. Confirms that the service is ready for predictions.
    4. Loads 5 random sample images from the training data CSV.
    5. For each sample, it predicts the product category from the image.
    6. Prints the true label, predicted label, and confidence score.
    7. Indicates whether the prediction was correct.
    """
    print("Testing CNN Product Detection Service...")
    
    # Check if model files exist
    model_path = 'models/product_classifier.h5'
    label_encoder_path = 'models/label_encoder.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first by running: python cnn_model_train.py")
        return
    
    if not os.path.exists(label_encoder_path):
        print(f"Label encoder file not found: {label_encoder_path}")
        print("Please train the model first by running: python cnn_model_train.py")
        return
    
    try:
        # Initialize the service
        cnn_service = CNNProductDetectionService()
        
        if not cnn_service.is_model_ready():
            print("CNN service is not ready")
            return
        
        print("CNN service initialized successfully!")
        
        # Load test data
        df = pd.read_csv('data/dataset/cnn_model_train.csv')
        
        # Test with a few sample images
        test_samples = df.sample(n=5, random_state=42)
        
        print("\nTesting with sample images:")
        print("-" * 50)
        
        for idx, row in test_samples.iterrows():
            img_path = row['image']
            true_label = row['description']
            
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        result = cnn_service.predict_product(f)
                    
                    predicted_class = result['predicted_class']
                    confidence = result['confidence']
                    
                    print(f"Image: {os.path.basename(img_path)}")
                    print(f"True label: {true_label}")
                    print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")
                    print(f"Correct: {'✓' if predicted_class == true_label else '✗'}")
                    print("-" * 30)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            else:
                print(f"Image not found: {img_path}")
        
        print("Testing completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_cnn_model()
