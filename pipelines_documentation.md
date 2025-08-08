# Pipelines Documentation

This document provides an overview of the data and model pipelines used in the application.

---

## 1. Recommendation Data Pipeline

**File:** `pipelines/recommendation_data_pipeline.py`

### Purpose
This pipeline is responsible for processing the product dataset, generating vector embeddings for product descriptions, and storing them in a vector knowledge base for efficient similarity searches.

### Steps
1.  **Load Data**: Loads the product dataset from `data/dataset/cleaned_dataset.csv`.
2.  **Clean and Filter**: Removes irrelevant or invalid data, such as entries with no description, a unit price of zero, or descriptions containing generic terms like 'ebay'.
3.  **Aggregate Products**: Groups the data by `StockCode` to create a unique list of products, calculating the mean `UnitPrice`.
4.  **Generate Embeddings**: Uses the `EmbeddingService` to create vector embeddings for each product description.
5.  **Prepare Metadata**: Structures the product information (stock code, description, unit price) into a metadata format.
6.  **Upsert to Vector Store**: Inserts or updates the embeddings and their corresponding metadata in the vector store using the `VectorService`.

### How to Run
This pipeline can be executed as a standalone script to populate or update the vector knowledge base:
```bash
python pipelines/recommendation_data_pipeline.py
```

---

## 2. CNN Training Pipeline

**File:** `pipelines/cnn_training_pipeline.py`

### Purpose
This pipeline orchestrates the training of a Convolutional Neural Network (CNN) for image-based product classification.

### Steps
1.  **Load Data**: Loads and preprocesses images and their corresponding labels from the training dataset using the `data_loader`.
2.  **Encode Labels**: Converts the text-based labels into one-hot encoded vectors suitable for training.
3.  **Split Data**: Splits the dataset into training and testing sets.
4.  **Create and Compile Model**: Builds a simple CNN model architecture and compiles it with an optimizer (`adam`) and a loss function (`categorical_crossentropy`).
5.  **Train Model**: Trains the CNN on the training data for a predefined number of epochs.
6.  **Evaluate Model**: Measures the model's performance on the test set.
7.  **Save Artifacts**: Saves the trained model (`product_classifier.h5`) and the label encoder (`label_encoder.pkl`) to the `models/` directory for later use in inference.

### How to Run
This pipeline can be run as a standalone script to train the CNN model from scratch:
```bash
python pipelines/cnn_training_pipeline.py
```

---

## 3. CNN Inference Pipeline

**File:** `pipelines/cnn_inference_pipeline.py`

### Purpose
This pipeline uses the pre-trained CNN model to perform inference on a given image, predicting the product category.

### Steps
1.  **Load Artifacts**: Initializes by loading the saved CNN model and the label encoder.
2.  **Preprocess Image**: Takes an input image, converts it to RGB, resizes it to the required dimensions, and normalizes the pixel values.
3.  **Make Prediction**: Feeds the preprocessed image into the model to get a prediction.
4.  **Format Output**: Returns a dictionary containing the predicted class, the confidence score, and the top 3 predictions with their respective confidence scores.

---

## 4. OCR Inference Pipeline

**File:** `pipelines/ocr_inference_pipeline.py`

### Purpose
This pipeline integrates Optical Character Recognition (OCR) with the product recommendation service. It extracts text from an image and uses it as a query to find relevant products.

### Dependencies
-   `ProductRecommendationService`: Required for fetching recommendations based on the extracted text.

### Steps
1.  **Extract Text**: Uses the `OCRService` to extract text and a confidence score from an input image.
2.  **Validate Text**: Checks if any text was found and if it's readable enough to be a valid query.
3.  **Clean Text**: Cleans the raw extracted text by removing common OCR artifacts and extra whitespace.
4.  **Get Recommendations**: Passes the cleaned text to the `ProductRecommendationService` to get a list of recommended products.
5.  **Format Output**: Returns a dictionary containing the extracted text, OCR confidence, recommended products, and the cleaned query.

This document explains the pipelines implemented in this codebase, why they exist, how they are used, and where they are invoked. It also references relevant source locations with line numbers for quick navigation.


## Overview: Why Pipelines?

Pipelines encapsulate multi-step workflows into cohesive, reusable units. They keep `app.py` thin and push orchestration complexity into components that are:

- Modular: Easy to test and maintain.
- Reusable: Called by services or scripts without duplicating logic.
- Replaceable: You can evolve pipeline internals without changing route wiring.

Current pipelines:
- `OcrInferencePipeline` – OCR to recommendations flow
- `CNNInferencePipeline` – CNN model loading and inference
- `CNNTrainingPipeline` – End-to-end training orchestration
- `RecommendationDataPipeline` – Vector store bootstrap from the products dataset

---

## Where Pipelines Are Used

### 1) Application Startup (Bootstrap)

File: `app.py`
- Import: lines 21–26 include imports of service layers and the data pipeline
  - `from pipelines.recommendation_data_pipeline import RecommendationDataPipeline` (line 26)
- Conditional pipeline run after initializing the recommendation service:
  - Lines 45–48: initialize `ProductRecommendationService`
  - Lines 50–61: check the vector index stats; if empty, run `RecommendationDataPipeline().run()` to populate Pinecone

Why here?
- `RecommendationDataPipeline` prepares the vector index used by `ProductRecommendationService`. Running it once at startup (only if empty) ensures the app can answer queries immediately without manual preloading.

### 2) OCR Query Flow (Runtime)

File: `services/ocr/ocr_query_service.py`
- Import pipeline: line 11 `from pipelines.ocr_inference_pipeline import OcrInferencePipeline`
- Instantiate pipeline: line 21 `self.pipeline = OcrInferencePipeline(recommendation_service)`
- Execute on request: lines 23–26 `return self.pipeline.run(image_data)`

How it’s used:
- The Flask route `/ocr-query` (in `app.py`) calls `OCRQueryService.process_image_query(...)`, which delegates to `OcrInferencePipeline` to extract text and fetch recommendations.

Why in the service (not app.py)?
- Keeps route handlers clean. The service owns domain logic; the pipeline owns orchestration.

### 3) Image Product Detection (Runtime)

File: `services/image_detection_service/product_detection_service.py`
- Import pipeline: line 11 `from pipelines.cnn_inference_pipeline import CNNInferencePipeline`
- Instantiate pipeline: lines 20–22 `self.pipeline = CNNInferencePipeline()`
- Execute on request: lines 28–33 `return self.pipeline.run(image_file)`

How it’s used:
- The Flask route handling image-based product search calls `CNNProductDetectionService.predict_product(...)`, which runs the inference pipeline to predict class and score(s).

### 4) Model Training (Offline)

File: `cnn_training/cnn_train.py`
- Import pipeline: line 23 `from pipelines.cnn_training_pipeline import CNNTrainingPipeline`
- Execute: lines 26–29 `pipeline = CNNTrainingPipeline(); pipeline.run()`

How it’s used:
- This is a command-line training entry point. It orchestrates data loading, model creation, training, evaluation, and artifact saving via the pipeline.

---

## Pipeline Details

### A) RecommendationDataPipeline
File: `pipelines/recommendation_data_pipeline.py`
- Class definition: line 16 `class RecommendationDataPipeline:`
- Run entry: line 24 `def run(self, data_path='data/dataset/cleaned_dataset.csv'):`
- Steps:
  - Load dataset (line 29)
  - Clean/filter rows (lines 31–38)
  - Aggregate unique products by `StockCode` (lines 40–44)
  - Batch embeddings for descriptions (lines 46–49)
  - Build metadata and upsert to Pinecone (lines 50–69)

Used in:
- App startup (bootstrap) in `app.py` lines 50–61.

Why:
- Ensures the vector database backing the recommendation system is populated with product embeddings and metadata.

### B) OcrInferencePipeline
File: `pipelines/ocr_inference_pipeline.py`
- Class definition: line 15 `class OcrInferencePipeline:`
- Purpose: extract text from an incoming image, clean it, and request recommendations from `ProductRecommendationService`.
- Called by: `OCRQueryService` (see lines 11, 21, 23–26 in `services/ocr/ocr_query_service.py`).

Why:
- Encapsulates OCR + text cleaning + recommendation lookup. Keeps service thin and focused on error handling and orchestration.

### C) CNNInferencePipeline
File: `pipelines/cnn_inference_pipeline.py`
- Class definition: line 19 `class CNNInferencePipeline:`
- Purpose: load the Keras model and label encoder, preprocess images, and perform prediction.
- Called by: `CNNProductDetectionService` (see lines 11, 20–22, 28–33 in `services/image_detection_service/product_detection_service.py`).

Why:
- Separates model and preprocessing details from the HTTP layer and business logic; reusability and testability.

### D) CNNTrainingPipeline
File: `pipelines/cnn_training_pipeline.py`
- Class definition: line 20 `class CNNTrainingPipeline:`
- Purpose: end-to-end training flow (data, model, training, evaluation, saving artifacts).
- Called by: `cnn_training/cnn_train.py` (lines 23, 26–29).

Why:
- Provides an offline training workflow decoupled from runtime app logic.

---

## Design Rationale: Why Services Call Pipelines (Not app.py)

- **Separation of concerns**: `app.py` focuses on routing and dependency wiring. Services encapsulate domain logic and coordinate pipelines.
- **Testability**: You can unit test services and pipelines independently without running the Flask app.
- **Flexibility**: Pipelines can change (e.g., different OCR library or model) without touching routes.
- **Operational clarity**: The only pipeline run in `app.py` is the bootstrap pipeline (`RecommendationDataPipeline`) to ensure the vector store is ready.

---

## Quick Reference: Source Locations

- `app.py`
  - Imports and bootstrap: lines 21–26, 45–61
- `services/ocr/ocr_query_service.py`
  - Pipeline usage: lines 11, 21, 23–26
- `services/image_detection_service/product_detection_service.py`
  - Pipeline usage: lines 11, 20–22, 28–33
- `cnn_training/cnn_train.py`
  - Pipeline usage: lines 23, 26–29
- `pipelines/recommendation_data_pipeline.py`
  - Pipeline steps: lines 16, 24, 29–69

---



