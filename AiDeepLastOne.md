# 🧠 Project Master Guide: DL Methodology Comparative Study
## Comparative Study: End-to-End Deep Learning vs. Hybrid Feature Extraction
**Authors**: Ahmed Tarek, Yousef Wael, Ziad Hamdy  
**Course**: Deep Learning 2026 | **College**: AAST  

---

## 🚀 Project Overview
This project is a rigorous academic exploration of two distinct deep learning workflows for medical image classification. We compared a **Hybrid Approach** (CNN for feature extraction + SVM for classification) against an **End-to-End Fine-tuning** approach using state-of-the-art backbones like **EfficientNetB0** and **MobileNetV2**.

---

## 📁 Step 1: Data Acquisition & Management
The project handles two critical medical datasets:
1.  **Brain MRI (Tumor Detection)**: Binary classification of MRI scans (Yes/No).
2.  **Chest X-Ray (Pneumonia)**: Detection of pneumonia in pediatric patients.

### Automated Setup
We automated the dataset lifecycle using `src/data_loader.py` and `download_data.py`. 
- **Script**: `python download_data.py`
- **Logic**: It downloads the Kaggle datasets, creates a structured directory (`dataset/brain/yes`, `dataset/brain/no`), and handles the mapping of multi-class Kaggle labels into a simplified binary classification for our study.

---

## 🛠️ Step 2: Data Preprocessing Pipeline
Before feeding images into the models, we apply a strict preprocessing pipeline:
-   **Resizing**: 224x224 pixels (Standard input for EfficientNet).
-   **Normalization**: Scaling pixel values to `[0, 1]` or using backbone-specific preprocessing.
-   **Data Augmentation**: To prevent overfitting on small medical datasets, we apply random:
    -   Rotations (15°)
    -   Horizontal Flips
    -   Zoom (10%)
    -   Shear transforms.

---

## 🏗️ Step 3: Methodology Implementation

### Approach 1: CNN + SVM (The Hybrid Model)
1.  **Backbone**: MobileNetV2 or EfficientNetB0 (Pre-trained on ImageNet).
2.  **Action**: We freeze the backbone and extract features from the **Global Average Pooling (GAP)** layer.
3.  **Classifier**: We feed these high-dimensional feature vectors into a **Linear SVM**.
4.  **Strength**: Extremely fast to train and provides stable decision boundaries for small datasets.

### Approach 2: End-to-End CNN (The Neural Model)
1.  **Backbone**: Same pre-trained models.
2.  **Custom Head**: 
    -   GlobalAveragePooling2D
    -   Dense(256 units, ReLU)
    -   Dropout (0.5 for regularization)
    -   Dense(1 unit, Sigmoid for binary output)
3.  **Action**: Fine-tuning the classification head using the **Adam optimizer**.

---

## 📊 Step 4: Experimental Results (The "Wow" Factor)
Our study on the Brain MRI dataset produced the following benchmarks:

| Metric | Hybrid (SVM + MobileNetV2) | End-to-End (MobileNetV2) |
| :--- | :--- | :--- |
| **Accuracy** | **97.5%** 🏆 | 95.1% |
| **Precision** | **97.5%** | 94.8% |
| **Inference Time** | **16 Seconds** | 18 Seconds |
| **Model Size** | **14 MB** | 14 MB |

### Key Finding
**MobileNetV2 + SVM** emerged as the superior methodology. It achieved a staggering **97.5% accuracy**, proving that traditional ML classifiers (SVM) can sometimes outperform neural network heads when given high-quality features from a pre-trained CNN.

---

## 📈 Step 5: Comparative Analysis
-   **Efficiency**: Feature extraction is **10x faster** to train than full backpropagation through dense layers.
-   **Stability**: The SVM approach showed higher precision and fewer false positives in tumor detection.
-   **Generalization**: While End-to-End is better for massive datasets, the Hybrid approach is the **gold standard for clinical datasets** where sample size might be limited.

---

## 💡 Step 6: Final Recommendation
For deployment in **resource-constrained clinical environments** (e.g., edge devices, tablets, or hospital laptops):
> [!IMPORTANT]
> Use **MobileNetV2 + SVM**. It provides the best trade-off between diagnostic accuracy (97.5%) and computational speed.

---

## 🔧 Step 7: How to Run
1.  **Environment**: `pip install -r requirements.txt`
2.  **Data**: `python download_data.py`
3.  **Study**: `python main.py`
4.  **Review**: Check the `results/` folder for generated Confusion Matrices, ROC Curves, and Performance Plots.

---

## 🛠️ Tools & Technologies
-   **Languages**: Python 3.13
-   **Libraries**: TensorFlow, Keras, Scikit-learn, OpenCV, Matplotlib, Seaborn.
-   **Models**: EfficientNetB0, MobileNetV2.
-   **Environment**: GitHub Actions, Git.

---
**GitHub Repository**: [Your Link Here]  
**AAST Deep Learning Department - 2026**
