import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from .model_factory import ModelFactory

class MLTrainer:
    def __init__(self, architecture='EfficientNetB0'):
        self.architecture = architecture
        self.base_model = ModelFactory.get_model(architecture, include_top=False)
        # Add GAP layer to base model for feature extraction
        output = GlobalAveragePooling2D()(self.base_model.output)
        self.feature_extractor = Model(inputs=self.base_model.input, outputs=output)

    def extract_features(self, X):
        print(f"Extracting features using {self.architecture}...")
        features = self.feature_extractor.predict(X, batch_size=32, verbose=1)
        return features

    def train_svm(self, X_train_features, y_train, X_test_features, y_test):
        print("Training SVM Classifier...")
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train_features, y_train)
        
        y_pred = clf.predict(X_test_features)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        acc = accuracy_score(y_test, y_pred)
        
        return clf, metrics, acc

    def train_logistic_regression(self, X_train_features, y_train, X_test_features, y_test):
        print("Training Logistic Regression Classifier...")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_features, y_train)
        
        y_pred = clf.predict(X_test_features)
        metrics = classification_report(y_test, y_pred, output_dict=True)
        acc = accuracy_score(y_test, y_pred)
        
        return clf, metrics, acc
