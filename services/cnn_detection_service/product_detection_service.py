"""
Service for product detection using a pre-trained CNN model.

This module defines the `CNNProductDetectionService` class, which is responsible
for loading a trained Keras model and a label encoder, preprocessing images,
and predicting product categories from those images.
"""
import os
import pickle
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
from PIL import Image

class CNNProductDetectionService:
    """Manages product detection using a CNN model.

    This service loads a trained Keras model and a corresponding label encoder
    to predict product categories from input images. It handles image preprocessing,
    prediction, and decoding of results.

    Attributes:
        model_path (str): The absolute path to the Keras model file (.h5).
        label_encoder_path (str): The absolute path to the pickled label encoder.
        model (keras.Model): The loaded Keras model.
        label_encoder (sklearn.preprocessing.LabelEncoder): The loaded label encoder.
        img_size (Tuple[int, int]): The target image dimensions for preprocessing.
    """
    def __init__(self, model_path: str = 'models/product_cnn_model.h5', 
                 label_encoder_path: str = 'models/label_encoder.pkl') -> None:
        """Initializes the CNNProductDetectionService.

        Constructs absolute paths for the model and label encoder files based on
        the project's root directory and attempts to load them.

        Args:
            model_path (str): The relative path to the Keras model file.
            label_encoder_path (str): The relative path to the label encoder file.
        """
        # Determine the absolute path to the project's root directory
        # The current file is in services/cnn_detection_service/, so we go up two levels.
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Construct absolute paths for the model and label encoder
        self.model_path = os.path.join(base_dir, model_path)
        self.label_encoder_path = os.path.join(base_dir, label_encoder_path)
        self.model = None
        self.label_encoder = None
        self.img_size = (64, 64)
        
        # Load model and label encoder
        self._load_model()
        self._load_label_encoder()
    
    def _load_model(self) -> None:
        """Loads the trained Keras CNN model from the specified path.

        Raises:
            FileNotFoundError: If the model file does not exist at the path.
            Exception: For any other errors during model loading.
        """
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"CNN model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            raise e
    
    def _load_label_encoder(self) -> None:
        """Loads the pickled label encoder from the specified path.

        Raises:
            FileNotFoundError: If the label encoder file does not exist.
            Exception: For any other errors during file loading or unpickling.
        """
        try:
            if os.path.exists(self.label_encoder_path):
                with open(self.label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"Label encoder loaded successfully from {self.label_encoder_path}")
            else:
                print(f"Label encoder file not found at {self.label_encoder_path}")
                raise FileNotFoundError(f"Label encoder file not found at {self.label_encoder_path}")
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            raise e
    
    def preprocess_image(self, image_file: Any) -> np.ndarray:
        """Preprocesses an image file for prediction.

        Opens the image, converts it to RGB, resizes it, normalizes the pixel
        values to [0, 1], and adds a batch dimension.

        Args:
            image_file: A file-like object representing the image.

        Returns:
            A numpy array representing the preprocessed image.

        Raises:
            Exception: If there is an error during image processing.
        """
        try:
            # Open and convert image
            img = Image.open(image_file)
            img = img.convert('RGB')
            img = img.resize(self.img_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise e
    
    def predict_product(self, image_file: Any) -> Dict[str, Any]:
        """Predicts the product category from an image file.

        Preprocesses the image, uses the CNN model to make a prediction, and
        decodes the result to get the predicted class, confidence score, and
        top-3 predictions.

        Args:
            image_file: A file-like object representing the image.

        Returns:
            A dictionary containing:
                - 'predicted_class' (str): The name of the top predicted class.
                - 'confidence' (float): The confidence score for the top prediction.
                - 'top_3_predictions' (List[Dict[str, Any]]): A list of the top 3
                  predictions, each with 'class' and 'confidence'.

        Raises:
            Exception: If there is an error during prediction.
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_file)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # Decode label
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            
            for idx in top_3_indices:
                class_name = self.label_encoder.inverse_transform([idx])[0]
                class_confidence = float(predictions[0][idx])
                top_3_predictions.append({
                    'class': class_name,
                    'confidence': class_confidence
                })
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_3_predictions': top_3_predictions
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise e
    
    def is_model_ready(self) -> bool:
        """Checks if the model and label encoder are successfully loaded.

        Returns:
            True if both the model and label encoder are loaded, False otherwise.
        """
        return self.model is not None and self.label_encoder is not None
