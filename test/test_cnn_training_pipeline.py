import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Mock dependencies before importing the pipeline
mock_config = MagicMock(
    RANDOM_STATE=42, 
    TEST_SPLIT_SIZE=0.2, 
    BATCH_SIZE=32, 
    EPOCHS=1, 
    MODEL_DIR='models', 
    MODEL_SAVE_PATH='models/model.h5', 
    LABEL_ENCODER_PATH='models/encoder.pkl'
)
mock_data_loader = MagicMock()
mock_cnn_model = MagicMock()
mock_keras_utils = MagicMock()
mock_label_encoder = MagicMock()
mock_train_test_split = MagicMock()

modules = {
    'cnn_training.config': mock_config,
    'cnn_training.data_loader': mock_data_loader,
    'cnn_training.cnn_model': mock_cnn_model,
    'keras.utils': mock_keras_utils,
    'sklearn.preprocessing': MagicMock(LabelEncoder=lambda: mock_label_encoder),
    'sklearn.model_selection': MagicMock(train_test_split=mock_train_test_split),
    'tensorflow': MagicMock(),
    'keras': MagicMock()
}

@patch.dict(sys.modules, modules)
class TestCNNTrainingPipeline(unittest.TestCase):
    """Test suite for the CNNTrainingPipeline."""

    def test_run_pipeline_orchestration(self):
        """Tests that the training pipeline calls all steps in the correct order."""
        # Late import to ensure mocks are applied
        from pipelines.cnn_training_pipeline import CNNTrainingPipeline

        # --- Mock setup ---
        # Mock data loader
        mock_images = np.random.rand(10, 50, 50, 3)
        mock_labels = ['cat'] * 5 + ['dog'] * 5
        mock_data_loader.load_and_preprocess_data.return_value = (mock_images, mock_labels)

        # Mock label encoder
        mock_label_encoder.classes_ = ['cat', 'dog']

        # Mock train_test_split
        mock_train_test_split.return_value = (1, 2, 3, 4) # Dummy values

        # Mock CNN model
        mock_model = MagicMock()
        mock_cnn_model.create_simple_cnn_model.return_value = mock_model

        # --- Execution ---
        with patch('builtins.open', mock_open()) as mock_file, patch('pickle.dump') as mock_pickle_dump:
            pipeline = CNNTrainingPipeline()
            pipeline.run()

        # --- Assertions ---
        # 1. Load data
        mock_data_loader.load_and_preprocess_data.assert_called_once()
        # 2. Encode labels
        mock_label_encoder.fit_transform.assert_called_with(mock_labels)
        # 3. Split data
        mock_train_test_split.assert_called_once()
        # 4. Create and compile model
        mock_cnn_model.create_simple_cnn_model.assert_called_once()
        mock_model.compile.assert_called_once()
        # 5. Train model
        mock_model.fit.assert_called_once()
        # 6. Evaluate model
        mock_model.evaluate.assert_called_once()
        # 7. Save model and encoder
        mock_model.save.assert_called_with(mock_config.MODEL_SAVE_PATH)
        mock_pickle_dump.assert_called_once()

if __name__ == '__main__':
    unittest.main()
