"""
Configuration settings for the CNN model training pipeline.

This module centralizes all the key parameters required for training the product
detection model, including file paths, image dimensions, and hyperparameters.
Keeping these settings in a single file makes it easy to manage and adjust the
training process.
"""

import os
from typing import Tuple

# =============================================================================
# PATHS
# =============================================================================
# Absolute path to the project root directory.
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Absolute path to the training data CSV file.
CSV_PATH: str = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'cnn_model_train.csv')

# Directory to save the trained model and label encoder.
MODEL_DIR: str = os.path.join(PROJECT_ROOT, 'models')

# Full path for saving the trained Keras model.
MODEL_SAVE_PATH: str = os.path.join(MODEL_DIR, 'product_classifier.h5')

# Full path for saving the label encoder.
LABEL_ENCODER_PATH: str = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# Alias for inference pipeline
MODEL_PATH: str = MODEL_SAVE_PATH
ENCODER_PATH: str = LABEL_ENCODER_PATH


# =============================================================================
# IMAGE & MODEL PARAMETERS
# =============================================================================
# Target image size for resizing (width, height).
IMG_SIZE: Tuple[int, int] = (64, 64)

# Unpack image dimensions for individual access.
IMAGE_WIDTH, IMAGE_HEIGHT = IMG_SIZE

# Number of samples per gradient update.
BATCH_SIZE: int = 32

# Number of times to iterate over the entire training dataset.
EPOCHS: int = 20

# Fraction of data to use for validation.
TEST_SPLIT_SIZE: float = 0.2

# Seed for random number generators to ensure reproducibility.
RANDOM_STATE: int = 42
