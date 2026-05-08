![Medical AI Comparative Study](file:///C:/Users/20100/.gemini/antigravity/brain/4d60d58f-c4be-485d-9991-7524f725babe/project_banner_1778161671386.png)

# Comparative Study: End-to-End DL vs. Feature Extraction with ML


## 🎓 University Project Overview
This project presents a comprehensive comparison between two mainstream approaches in medical image classification:
1. **Approach 1**: Using a pre-trained CNN as a fixed feature extractor followed by a traditional Machine Learning classifier (SVM).
2. **Approach 2**: Training a full end-to-end Deep Learning model using fine-tuning.

### 🔬 Key Features
- **Architectures**: EfficientNetB0 (Primary), MobileNetV2 (Bonus).
- **Datasets**: Brain MRI Tumor Detection, Chest X-Ray Pneumonia (Bonus).
- **Frameworks**: TensorFlow, Keras, Scikit-learn, OpenCV.
- **Evaluation**: Professional confusion matrices, learning curves, and comparative bar charts.

## 📁 Project Structure & File Descriptions

### 🚀 Core Execution
*   **`main.py`**: The "engine" of the project. It automates the entire experiment suite. It iterates through different datasets (Brain Tumor, Pneumonia) and architectures (EfficientNet, MobileNet), executes both the Hybrid (SVM) and End-to-End (CNN) approaches, saves the results, and generates final comparison charts.

### 📦 Source Code (`src/`)
*   **`data_loader.py`**: Handles downloading, organizing, resizing (224x224), normalizing, and augmenting image data.
*   **`model_factory.py`**: Defines the Deep Learning architectures and builds the End-to-End models with specific hyperparameters (Adam, 1e-4 LR).
*   **`feature_extractor.py`**: Manages the extraction of high-level features from CNNs and training of ML classifiers like **SVM**.
*   **`evaluator.py`**: The visualization tool for creating Confusion Matrices, Learning Curves, and Bar Charts.
*   **`utils.py`**: Helper functions for logging progress and managing folders.

### 📊 Results & Output
*   **`results/`**: Stores all generated plots (`.png`) and the final `full_comparison_metrics.csv`.
*   **`presentation/`**: Contains the `presentation.md` file for your project defense.
*   **`dataset/`**: Stores the raw and processed image data.

## 🚀 Getting Started
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd project-root
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   The project uses Kaggle datasets. Ensure you have the `kaggle.json` credentials or manually place data in `dataset/brain_tumor/`.

4. **Run the Study**:
   ```bash
   python main.py
   ```

## 📊 Results & Visualization
Experiments successfully completed!
- **Top Performer**: **MobileNetV2 + SVM** (97.5% Accuracy)
- **EfficientNetB0 (E2E)**: 75% Accuracy

All plots, including confusion matrices and learning curves, are available in the `results/` folder.

## 📝 Authors
**Students**: Ahmed Tarek, Yousef Wael, Ziad Hamdy  
**College**: AAST (Arab Academy for Science, Technology & Maritime Transport)
