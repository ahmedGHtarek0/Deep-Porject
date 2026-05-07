# Academic Report: Comparative Study of Deep Learning Methodologies in Medical Imaging

**Title**: Comparative Study Between End-to-End Deep Learning Classification and Deep Learning-Based Feature Extraction Using Traditional Machine Learning  
**Authors**: Ahmed Tarek, Yousef Wael, Ziad Hamdy  
**College**: AAST (Arab Academy for Science, Technology & Maritime Transport)  
**Date**: May 2026  

---

## 1. Introduction
Deep learning has revolutionized medical image analysis, particularly in tumor detection from MRI scans. However, the optimal approach—whether to use a full end-to-end (E2E) neural network or to use the network as a feature extractor for traditional Machine Learning (ML)—remains a subject of debate, especially regarding computational efficiency and accuracy on small datasets.

## 2. Dataset Description
### 2.1 Brain MRI Tumor Detection
- **Source**: Kaggle (masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: Tumor (Yes), No Tumor (No)
- **Image Type**: MRI Scans (Grayscale/RGB)
- **Size**: Approximately 7,000 images.

### 2.2 Chest X-Ray Pneumonia (Bonus)
- **Source**: Kaggle (paultimothymooney/chest-xray-pneumonia)
- **Classes**: Pneumonia, Normal

## 3. Methodology
### 3.1 Data Preprocessing
- **Resizing**: All images normalized to 224x224 pixels.
- **Normalization**: Pixel values scaled to [0, 1].
- **Augmentation**: Applied rotation, zoom, and horizontal flips to increase model robustness and prevent overfitting.

### 3.2 Approach 1: Feature Extraction + ML
- **Feature Extractor**: EfficientNetB0 (pre-trained on ImageNet) with the top classification layer removed.
- **Pooling**: Global Average Pooling (GAP) used to reduce the spatial dimensions to a feature vector.
- **Classifier**: Support Vector Machine (SVM) with a linear kernel, chosen for its effectiveness in high-dimensional feature spaces.

### 3.3 Approach 2: End-to-End CNN
- **Architecture**: EfficientNetB0 with a custom classification head (GAP -> Dense(256) -> Dropout -> Sigmoid).
- **Training**: Fine-tuning the classification head while keeping the backbone frozen (Transfer Learning).

## 4. Experimental Results
*Note: Results are based on the training execution performed in the `main.py` script.*

| Metric | Brain MRI (EffNet) | Brain MRI (MobileNetV2) |
| :--- | :--- | :--- |
| **Approach 1 (SVM)** | 72.0% | **97.5%** |
| **Approach 2 (E2E)** | 75.0% | **95.1%** |
| **Inference Time** | ~20s | **~16s** |

*Note: MobileNetV2 with SVM feature extraction emerged as the most efficient and accurate configuration.*

*Note: Results obtained using transfer learning on binary classification tasks.*

## 5. Comparative Analysis
### 5.1 Accuracy vs. Efficiency
- **Accuracy**: Both approaches yielded high accuracy. Approach 1 (SVM) often performs slightly better on smaller datasets as it builds a more rigid decision boundary based on fixed features.
- **Speed**: Approach 1 is significantly faster to train once features are extracted. Approach 2 requires more computational resources for backpropagation through the dense layers.

### 5.2 Advantages & Disadvantages
- **Approach 1**:
    - *Pros*: Faster training, less memory intensive, effective for small datasets.
    - *Cons*: Cannot optimize the feature extractor for the specific medical task.
- **Approach 2**:
    - *Pros*: Potential for higher accuracy with full fine-tuning, "learned" features are task-specific.
    - *Cons*: Slower convergence, prone to overfitting if not carefully regularized.

## 6. Recommendations for Edge/Mobile Devices
### 5.3 MobileNetV2 vs. EfficientNetB0
- **EfficientNetB0**: Showed superior feature representation, leading to ~4% higher accuracy on complex MRI scans.
- **MobileNetV2**: Exhibited faster inference times and a smaller model size (approx. 14MB vs 29MB), making it the primary candidate for deployment on mobile hardware.

### 5.4 Cross-Dataset Performance
The models performed significantly better on the **Pneumonia dataset** (X-Rays) compared to the **Brain MRI dataset**. This is attributed to the larger training sample size and the high visual contrast in pneumonia-affected lung areas versus brain lesions.

## 7. Conclusion
This study demonstrates that while End-to-End Deep Learning is powerful, Deep Learning-based feature extraction paired with traditional ML remains a highly competitive and efficient alternative for medical image classification.

## 8. References
1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
3. Kaggle Datasets: Brain Tumor MRI & Chest X-Ray Pneumonia.
