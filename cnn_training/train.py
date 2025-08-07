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
import sys
import os

# Add the project root (one level up) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from cnn_training import config
from cnn_training import data_loader
from cnn_training import model

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

def main() -> None:
    """
    Main function to orchestrate the training pipeline.

    This function handles the creation of the model output directory, loading and
    preprocessing of the data, label encoding, data splitting, model creation,
    compilation, training, evaluation, and finally, saving the trained model
    and label encoder.
    """
    # Create models directory if it doesn't exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Load and preprocess data
    images, labels = data_loader.load_and_preprocess_data()
    
    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Convert to categorical
    encoded_labels = keras.utils.to_categorical(encoded_labels, num_classes)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, encoded_labels, 
        test_size=config.TEST_SPLIT_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=encoded_labels
    )
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    
    # Create model
    print("Creating CNN model...")
    cnn_model = model.create_simple_cnn_model(num_classes)
    
    # Compile model
    cnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    cnn_model.summary()
    
    # Train model
    print("Training model...")
    history = cnn_model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model and label encoder
    print("Saving model...")
    cnn_model.save(config.MODEL_SAVE_PATH)
    
    print("Saving label encoder...")
    with open(config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Training completed successfully!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Label encoder saved to: {config.LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    # This block ensures that the main function is called only when the script
    # is executed directly, not when it's imported as a module.
    main()