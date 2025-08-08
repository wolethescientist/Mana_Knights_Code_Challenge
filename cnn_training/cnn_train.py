"""
Main training script for the CNN model.

This script orchestrates the entire model training pipeline, including:
1. Setting up the Python path.
2. Loading and preprocessing data from the specified CSV file.
3. Encoding the labels.
4. Splitting the data into training and validation sets.
5. Creating and compiling the CNN model.
6. Training the model.
7. Evaluating the model's performance.
8. Saving the trained model and the label encoder to disk.
"""
# cnn_training/cnn_train.py

import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from pipelines.cnn_training_pipeline import CNNTrainingPipeline


def main():
    """Initializes and runs the CNN training pipeline."""
    pipeline = CNNTrainingPipeline()
    pipeline.run()


if __name__ == '__main__':
    main()