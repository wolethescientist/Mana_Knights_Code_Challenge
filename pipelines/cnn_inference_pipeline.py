# pipelines/cnn_inference_pipeline.py

import os
import sys
import pickle
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image
from keras.models import load_model

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from cnn_training import config


class CNNInferencePipeline:
    """Pipeline for CNN model inference, including model loading and prediction."""

    def __init__(self, model_path=config.MODEL_PATH, encoder_path=config.ENCODER_PATH):
        """Initializes the pipeline by loading the model and label encoder."""
        self.model = load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def _preprocess_image(self, image_data: Any) -> np.ndarray:
        """Preprocesses the input image for model prediction."""
        image = Image.open(image_data).convert('RGB')
        image = image.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)

    def run(self, image_data: Any) -> Dict[str, Any]:
        """Executes the inference pipeline for a given image."""
        # 1. Preprocess the image
        processed_image = self._preprocess_image(image_data)

        # 2. Make a prediction
        predictions = self.model.predict(processed_image)[0]

        # 3. Get top prediction
        top_prediction_index = np.argmax(predictions)
        confidence = predictions[top_prediction_index]
        predicted_class = self.label_encoder.classes_[top_prediction_index]

        # 4. Get top-3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            {
                'class': self.label_encoder.classes_[i],
                'confidence': float(predictions[i])
            }
            for i in top_3_indices
        ]

        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions
        }


