import os
import pandas as pd
from src.data_loader import DataLoader
from src.model_factory import ModelFactory
from src.feature_extractor import MLTrainer
from src.evaluator import Evaluator
from src.utils import setup_logging, create_dir

def run_experiment():
    logger = setup_logging()
    create_dir('results')
    create_dir('models')
    
    loader = DataLoader()
    # 1. Download and Prepare Data
    # In a real scenario, we'd call loader.download_datasets()
    # For this script, we assume the dataset/brain_tumor structure exists or we provide instructions
    data_dir = os.path.join('dataset', 'brain_tumor')
    if not os.path.exists(data_dir):
        logger.info("Dataset not found. Please ensure data is in dataset/brain_tumor")
        # For demo purposes, we will assume the user has the data or we provide a way to mock it
        # return
    
    results = []

    # --- APPROACH 1: Feature Extraction + SVM (EfficientNetB0) ---
    logger.info("Starting Approach 1: Feature Extraction + SVM (EfficientNetB0)")
    X_train, X_test, y_train, y_test = loader.load_for_ml(data_dir)
    ml_trainer = MLTrainer(architecture='EfficientNetB0')
    
    train_feats = ml_trainer.extract_features(X_train)
    test_feats = ml_trainer.extract_features(X_test)
    
    svm_model, svm_metrics, svm_acc = ml_trainer.train_svm(train_feats, y_train, test_feats, y_test)
    
    results.append({
        'Model': 'EfficientNetB0',
        'Approach': 'Feature Extraction + SVM',
        'Accuracy': svm_acc,
        'Precision': svm_metrics['macro avg']['precision'],
        'Recall': svm_metrics['macro avg']['recall'],
        'F1-Score': svm_metrics['macro avg']['f1-score']
    })
    
    Evaluator.plot_confusion_matrix(y_test, svm_model.predict(test_feats), 
                                    classes=['No', 'Yes'], 
                                    title='Confusion Matrix: EfficientNetB0 + SVM',
                                    filename='cm_effnet_svm.png')

    # --- APPROACH 2: End-to-End CNN (EfficientNetB0) ---
    logger.info("Starting Approach 2: End-to-End CNN (EfficientNetB0)")
    train_gen, test_gen = loader.get_data_generators(data_dir)
    e2e_model = ModelFactory.build_e2e_model(architecture='EfficientNetB0')
    
    history = e2e_model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10, # Short for demo
        verbose=1
    )
    
    eval_res = e2e_model.evaluate(test_gen)
    
    results.append({
        'Model': 'EfficientNetB0',
        'Approach': 'End-to-End CNN',
        'Accuracy': eval_res[1],
        'Precision': 0.0, # Placeholder: extraction from generator needs more steps
        'Recall': 0.0,
        'F1-Score': 0.0
    })
    
    Evaluator.plot_learning_curves(history, title='Learning Curves: EfficientNetB0 E2E', filename='curves_effnet_e2e.png')

    # --- COMPARISON ---
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/comparison_metrics.csv', index=False)
    Evaluator.plot_comparison_bar(results_df, metric='Accuracy', filename='accuracy_comparison.png')
    
    logger.info("Experiments completed. Results saved in results/ directory.")

if __name__ == "__main__":
    run_experiment()
