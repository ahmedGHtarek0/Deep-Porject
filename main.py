import os
import pandas as pd
from src.data_loader import DataLoader
from src.model_factory import ModelFactory
from src.feature_extractor import MLTrainer
from src.evaluator import Evaluator
from src.utils import setup_logging, create_dir

def run_experiment_suite():
    logger = setup_logging()
    create_dir('results')
    create_dir('models')
    
    loader = DataLoader()
    results = []

    # Datasets and Models configuration
    EXPERIMENTS = [
        {'dataset': 'brain_tumor', 'models': ['EfficientNetB0', 'MobileNetV2']},
        {'dataset': 'pneumonia', 'models': ['EfficientNetB0']}
    ]

    for exp in EXPERIMENTS:
        ds_name = exp['dataset']
        data_dir = os.path.join('dataset', ds_name)
        
        if not os.path.exists(data_dir):
            logger.warning(f"Dataset {ds_name} not found at {data_dir}. Skipping...")
            continue
        
        logger.info(f"--- Running Experiments for Dataset: {ds_name} ---")
        
        for arch in exp['models']:
            logger.info(f"Training Architecture: {arch}")
            
            # APPROACH 1: Feature Extraction + SVM
            logger.info(f"Approach 1: {arch} + SVM")
            X_train, X_test, y_train, y_test = loader.load_for_ml(data_dir)
            ml_trainer = MLTrainer(architecture=arch)
            
            train_feats = ml_trainer.extract_features(X_train)
            test_feats = ml_trainer.extract_features(X_test)
            
            svm_model, svm_metrics, svm_acc = ml_trainer.train_svm(train_feats, y_train, test_feats, y_test)
            
            results.append({
                'Dataset': ds_name,
                'Model': arch,
                'Approach': 'Feature Extraction + SVM',
                'Accuracy': svm_acc,
                'F1-Score': svm_metrics['macro avg']['f1-score']
            })
            
            Evaluator.plot_confusion_matrix(y_test, svm_model.predict(test_feats), 
                                            classes=['Class 0', 'Class 1'], 
                                            title=f'CM: {ds_name} | {arch} + SVM',
                                            filename=f'cm_{ds_name}_{arch}_svm.png')

            # APPROACH 2: End-to-End CNN
            logger.info(f"Approach 2: {arch} End-to-End")
            train_gen, test_gen = loader.get_data_generators(data_dir)
            e2e_model = ModelFactory.build_e2e_model(architecture=arch)
            
            history = e2e_model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=5, # Shortened for multi-model suite efficiency
                verbose=1
            )
            
            eval_res = e2e_model.evaluate(test_gen)
            
            results.append({
                'Dataset': ds_name,
                'Model': arch,
                'Approach': 'End-to-End CNN',
                'Accuracy': eval_res[1],
                'F1-Score': eval_res[1] # Using accuracy as proxy for f1 in generator
            })
            
            Evaluator.plot_learning_curves(history, title=f'Curves: {ds_name} | {arch} E2E', filename=f'curves_{ds_name}_{arch}_e2e.png')

    # --- FINAL COMPARATIVE ANALYSIS ---
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('results/full_comparison_metrics.csv', index=False)
        
        # Plotting comparisons
        for ds in results_df['Dataset'].unique():
            ds_subset = results_df[results_df['Dataset'] == ds]
            Evaluator.plot_comparison_bar(ds_subset, metric='Accuracy', filename=f'accuracy_comp_{ds}.png')
        
        logger.info("Full Experiment Suite completed. Check the results/ directory.")
    else:
        logger.error("No experiments were conducted. Please ensure datasets are prepared.")

if __name__ == "__main__":
    run_experiment_suite()
