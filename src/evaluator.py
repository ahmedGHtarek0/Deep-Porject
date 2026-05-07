import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from .utils import save_plot

class Evaluator:
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', filename='cm.png'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="white")
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title, fontsize=15, pad=20)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        save_plot(plt, filename)

    @staticmethod
    def plot_learning_curves(history, title='Learning Curves', filename='learning_curves.png'):
        plt.figure(figsize=(12, 5))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='#1f77b4')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, color='#ff7f0e')
        plt.title('Accuracy over Epochs', fontsize=13)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss', linewidth=2, color='#d62728')
        plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#2ca02c')
        plt.title('Loss over Epochs', fontsize=13)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.suptitle(title, fontsize=16)
        save_plot(plt, filename)

    @staticmethod
    def plot_comparison_bar(results_df, metric='Accuracy', filename='comparison.png'):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x='Approach', y=metric, hue='Model', data=results_df, palette='viridis')
        plt.title(f'Comparison of {metric} across Approaches', fontsize=15, pad=20)
        plt.ylim(0, 1.1)
        
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
        
        save_plot(plt, filename)

    @staticmethod
    def generate_report_table(results_list):
        df = pd.DataFrame(results_list)
        return df
