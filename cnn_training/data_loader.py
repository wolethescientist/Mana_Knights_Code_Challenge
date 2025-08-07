"""
Data loading and preprocessing for the CNN model.

This script is responsible for reading the dataset CSV, locating the image files,
and preparing them for training. It handles image loading, resizing, normalization,
and pairs each image with its corresponding label.
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

from cnn_training import config


def load_and_preprocess_data(
    csv_path: str = config.CSV_PATH, img_size: Tuple[int, int] = config.IMG_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads image paths from a CSV, preprocesses the images, and returns them as numpy arrays.

    This function reads a CSV file to get image paths and labels. It constructs
    absolute paths to the images, loads them, converts them to RGB, resizes them,
    and normalizes the pixel values to the [0, 1] range.

    Args:
        csv_path: The absolute path to the CSV file containing image paths and labels.
        img_size: A tuple representing the target size (width, height) for the images.

    Returns:
        A tuple containing two numpy arrays:
        - An array of preprocessed images.
        - An array of corresponding labels.
    """
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path)

    images: List[np.ndarray] = []
    labels: List[str] = []

    # The project root is the directory containing the 'data' folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(csv_path), '..', '..'))

    print(f"Processing {len(df)} images...")
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(df)} images")

        # Construct the full image path
        img_path = os.path.join(project_root, row['image'])
        label = row['description']

        try:
            # Check if file exists
            if not os.path.exists(img_path):
                print(f"File not found: {img_path}")
                continue

            # Load and preprocess image
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # Normalize to [0,1]

            images.append(img_array)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(labels)
