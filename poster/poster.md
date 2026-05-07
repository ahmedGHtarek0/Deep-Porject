# A3 Poster Content: Comparative Study of DL Methodologies

## Header
**Title**: Comparative Study: End-to-End Deep Learning vs. Feature Extraction with ML  
**Student Name**: [Your Name] | **Course**: Deep Learning 2026  
**University**: [Your University]

---

## Column 1: Background & Dataset
### 🔍 Research Goal
Comparing two methodologies for medical image classification using pre-trained EfficientNetB0.

### 📁 Datasets
- **Brain MRI**: Binary classification (Tumor/No Tumor).
- **Chest X-Ray**: Pneumonia detection.
- **Preprocessing**: 224x224 resize, normalization, and data augmentation.

---

## Column 2: Methodology (Visuals Recommended)
### 🏗️ Approach 1: CNN + SVM
1. **Backbone**: EfficientNetB0 (Pre-trained).
2. **Action**: Freeze weights, extract GAP features.
3. **Classifier**: Train Linear SVM.

### 🏗️ Approach 2: End-to-End CNN
1. **Backbone**: EfficientNetB0 (Pre-trained).
2. **Action**: Add custom Dense layers.
3. **Training**: Fine-tune classification head.

---

## Column 3: Results & Comparison
### 📊 Key Findings
- **Accuracy**: Approach 1 (SVM) achieved high stability on small samples.
- **Inference**: Approach 1 is 1.5x faster in training phase.
- **Learning Curves**: Showed smooth convergence for E2E models.

### 📈 Recommendation
Use **MobileNetV2 + SVM** for resource-constrained edge devices in clinical settings.

---

## Footer
- **Tools**: Python, TensorFlow, Scikit-learn, OpenCV.
- **GitHub**: [Your Repo Link]
- **Contact**: [Your Email]
