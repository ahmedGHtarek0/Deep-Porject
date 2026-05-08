# 🧠 Source Code Documentation (`src/`)

This folder contains the core logic for the Deep Learning comparative study. Below is a breakdown of each file and its responsibility:

---

### 1. `data_loader.py`
**Purpose:** Handles all data-related operations.
*   **Downloading:** Uses `kagglehub` to fetch datasets (Brain Tumor MRI and Chest X-Ray).
*   **Organization:** Reorganizes raw Kaggle data into standardized `yes/no` or class-based folders.
*   **Preprocessing:** 
    *   Resizes images to **224x224**.
    *   Normalizes pixel values to **[0, 1]**.
*   **Augmentation:** Applies real-time transformations (rotation, zoom, horizontal flip) to the training data to prevent overfitting.
*   **Splitting:** Splits the data into **Training (80%)** and **Validation (20%)** sets.

### 2. `model_factory.py`
**Purpose:** Manages the creation of Deep Learning models.
*   **Architectures:** Provides access to **EfficientNetB0** and **MobileNetV2**.
*   **E2E Building:** Contains the `build_e2e_model` function which:
    *   Loads pre-trained weights from ImageNet.
    *   Freezes the base layers (Transfer Learning).
    *   Adds custom Global Average Pooling, Dense, and Dropout layers.
    *   Compiles the model with the **Adam** optimizer and a learning rate of **1e-4**.

### 3. `feature_extractor.py`
**Purpose:** Implements the hybrid CNN + ML approach.
*   **Feature Extraction:** Uses the pre-trained base models (without the top layers) to convert images into high-dimensional numerical vectors (features).
*   **ML Classifiers:** Contains logic to train and evaluate:
    *   **SVM (Support Vector Machine):** Uses a linear kernel for classification.
    *   **Logistic Regression:** An alternative lightweight classifier.

### 4. `evaluator.py`
**Purpose:** Visualization and performance analysis.
*   **Confusion Matrix:** Generates `cm_..._.png` plots to show true vs. predicted labels.
*   **Learning Curves:** Plots training vs. validation Accuracy and Loss over epochs.
*   **Comparison Charts:** Creates bar charts comparing different models and approaches.

### 5. `utils.py`
**Purpose:** General helper functions.
*   **Logging:** Sets up professional console logging to track the training progress.
*   **Directory Management:** Automatically creates `results/` and `models/` folders if they don't exist.

### 6. `__init__.py`
**Purpose:** Makes the `src` folder a Python package, allowing you to import these modules into `main.py` using `from src.data_loader import DataLoader`.
