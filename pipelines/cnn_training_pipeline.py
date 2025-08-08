# pipelines/cnn_training_pipeline.py

import os
import pickle
import sys

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from cnn_training import config, cnn_model, data_loader


class CNNTrainingPipeline:
    """Orchestrates the CNN model training process."""

    def __init__(self):
        """Initializes the pipeline and sets random seeds for reproducibility."""
        np.random.seed(config.RANDOM_STATE)
        tf.random.set_seed(config.RANDOM_STATE)

    def run(self):
        """Executes the full training pipeline."""
        print("Starting CNN training pipeline...")

        # 1. Load and preprocess data
        images, labels = data_loader.load_and_preprocess_data()

        # 2. Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        categorical_labels = keras.utils.to_categorical(encoded_labels, num_classes)

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, categorical_labels, 
            test_size=config.TEST_SPLIT_SIZE, 
            random_state=config.RANDOM_STATE, 
            stratify=categorical_labels
        )

        # 4. Create and compile model
        model = cnn_model.create_simple_cnn_model(num_classes)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

        # 5. Train model
        model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # 6. Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # 7. Save model and label encoder
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        model.save(config.MODEL_SAVE_PATH)
        with open(config.LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder, f)

        print("CNN training pipeline completed successfully!")


if __name__ == '__main__':
    pipeline = CNNTrainingPipeline()
    pipeline.run()
