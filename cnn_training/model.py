"""
Defines the architecture for the Convolutional Neural Network (CNN).

This script contains the function responsible for creating the Keras model used
for product image classification. The architecture is a simple, standard CNN
with convolutional layers, pooling, and dense layers, designed for effective
feature extraction and classification.
"""

from typing import Tuple

import keras
from keras import layers, Model

from cnn_training import config


def create_simple_cnn_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (config.IMG_SIZE[0], config.IMG_SIZE[1], 3),
) -> Model:
    """Creates and returns a simple Convolutional Neural Network (CNN) model.

    The architecture consists of three convolutional blocks followed by a dense
    classifier head. Each convolutional block contains a Conv2D layer followed by
    MaxPooling2D. Dropout is used to prevent overfitting.

    Args:
        num_classes: The number of output classes for the final dense layer.
        input_shape: A tuple defining the shape of the input images (height, width, channels).

    Returns:
        A compiled Keras Sequential model.
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
