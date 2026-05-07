import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_plot(plt_obj, filename, folder='results'):
    create_dir(folder)
    path = os.path.join(folder, filename)
    plt_obj.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {path}")
