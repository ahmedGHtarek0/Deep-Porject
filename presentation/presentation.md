# Comparative Study: Brain Tumor Classification
## End-to-End Deep Learning vs. Hybrid CNN-SVM Approach

---

### 1. Project Overview
This project evaluates two state-of-the-art Deep Learning architectures (**EfficientNetB0** and **MobileNetV2**) on the **Brain Tumor MRI dataset**. We compared two distinct methodologies:
1.  **End-to-End CNN**: A fully automated deep learning pipeline.
2.  **Hybrid Approach**: CNN-based Feature Extraction followed by a Support Vector Machine (SVM) classifier.

---

### 2. Data Pipeline
#### Preprocessing & Specifications
*   **Image Resolution**: 224 x 224 pixels.
*   **Color Channels**: RGB (3 channels).
*   **Normalization**: Pixel values rescaled to the [0, 1] range.
*   **Classes**: Binary Classification (Tumor: Yes / No).

#### Data Augmentation
To improve model generalization and prevent overfitting, we applied:
*   **Rotation**: up to 20 degrees.
*   **Zoom**: up to 15%.
*   **Horizontal Flips**.

#### Data Splitting
*   **Training Set**: 80%
*   **Validation/Test Set**: 20%

---

### 3. Technical Justification of Models

#### EfficientNetB0
EfficientNet uses a **Compound Scaling** method that uniformly scales network width, depth, and resolution. This ensures that the model remains lightweight while achieving high accuracy.
> *Ref: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*

#### MobileNetV2
Designed for mobile and resource-constrained environments, MobileNetV2 introduces **Inverted Residuals** and **Linear Bottlenecks**. This architecture significantly reduces the number of parameters and operations without losing significant accuracy.
> *Ref: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.*

---

### 4. Implementation Details & Hyperparameters

| Hyperparameter | DL Implementation | ML Implementation (SVM) |
| :--- | :--- | :--- |
| **Optimizer** | Adam | N/A |
| **Learning Rate** | 1e-4 | N/A |
| **Batch Size** | 32 | 32 |
| **Epochs** | 5 | N/A |
| **Kernel Type** | N/A | Linear |
| **Regularization** | Dropout (0.5) | C=1.0 |

---

### 5. Quantitative Results

| Model | Approach | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| **EfficientNetB0** | Feature Extraction + SVM | 0.72 | 0.696 |
| **EfficientNetB0** | End-to-End CNN | 0.75 | 0.750 |
| **MobileNetV2** | Feature Extraction + SVM | **0.975** | **0.975** |
| **MobileNetV2** | End-to-End CNN | 0.951 | 0.951 |

---

### 6. Conclusion
*   **MobileNetV2** significantly outperformed EfficientNetB0 on this specific dataset, reaching **97.5% accuracy**.
*   The **Hybrid CNN-SVM approach** proved highly effective with MobileNetV2 features, suggesting that the extracted features are linearly separable.
*   End-to-End training is more robust for EfficientNetB0, but requires more fine-tuning for optimal performance.
