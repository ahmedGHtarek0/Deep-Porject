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

## 📁 Project Structure
```text
├── dataset/             # Raw and processed datasets
├── models/              # Saved model weights
├── results/             # Plots and metrics
├── report/              # Professional academic report
├── src/                 # Modular source code
│   ├── data_loader.py   # Preprocessing & augmentation
│   ├── model_factory.py # CNN architectures
│   └── evaluator.py     # Visualization tools
├── main.py              # Main execution script
└── requirements.txt     # Dependencies
```

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
