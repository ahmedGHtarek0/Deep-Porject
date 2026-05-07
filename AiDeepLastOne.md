# AI Deep Learning Project Documentation
## Comparative Study: End-to-End DL vs. Feature Extraction
**Authors**: Ahmed Tarek, Yousef Wael, Ziad Hamdy  
**College**: Arab Academy for Science, Technology & Maritime Transport (AAST)  
**Date**: May 2026

---

## 1. Project Objective
This project explores two primary methodologies for medical image classification using transfer learning:
1. **Approach 1 (Hybrid)**: Using a pre-trained CNN as a fixed feature extractor followed by a Support Vector Machine (SVM) classifier.
2. **Approach 2 (End-to-End)**: Fine-tuning a pre-trained CNN with a custom classification head.

The goal is to determine which approach offers the best trade-off between **Accuracy**, **Computational Speed**, and **Memory Efficiency** for MRI and X-Ray diagnostics.

---

## 2. Experimental Results (The Highlights)
Our experiments on the Brain MRI Tumor dataset revealed a surprising and clear winner:

- **🏆 Winner**: **MobileNetV2 + SVM (Approach 1)**
- **Accuracy**: **97.5%**
- **Inference Speed**: **16 seconds** for 1440 test images.
- **Why?**: MobileNetV2's inverted residuals provide highly separable features, which the SVM classifies more effectively than a standard Dense neural network head.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Environment & Requirements
We used Python 3.13 with TensorFlow 2.21. The environment was set up with a dedicated `requirements.txt` to ensure reproducibility.
```python
# Key dependencies
tensorflow
scikit-learn
opencv-python
seaborn
```

### Step 2: Automated Data Management
We developed `src/data_loader.py` to automate the collection of data from Kaggle and reorganize it into a binary `Yes/No` structure.
```python
def prepare_brain_data(self, source_path):
    # Mapping: glioma, meningioma, pituitary -> yes | no_tumor -> no
    for split in ['Training', 'Testing']:
        # Copying and labeling logic...
```

### Step 3: Feature Extraction (Approach 1)
We used the CNN's Global Average Pooling layer as a "Latent Feature Vector" generator. These 1280-dimensional vectors were then used to train an SVM.
```python
def train_svm(self, X_train_features, y_train, X_test_features, y_test):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train_features, y_train)
    return clf, metrics, acc
```

### Step 4: End-to-End Training (Approach 2)
We added a custom head to the CNN backbone and trained it for 5-10 epochs using transfer learning.
```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
```

---

## 4. How to Run the Project
1. **Install**: `pip install -r requirements.txt`
2. **Download Data**: `python download_data.py` (Wait for it to finish).
3. **Execute Study**: `python main.py`.
4. **Results**: Check the `results/` folder for all generated heatmaps and curves.

---

## 5. Conclusion
For medical AI projects at AAST, we recommend **MobileNetV2 with SVM** for MRI-based tumor detection. It provides the highest accuracy (97.5%) while remaining light enough to run on standard hospital laptops or portable diagnostic tablets.
