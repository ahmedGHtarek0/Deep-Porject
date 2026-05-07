import os
import shutil
from src.data_loader import DataLoader

def manual_reorganize():
    source = r"C:\Users\20100\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2"
    loader = DataLoader()
    print(f"Reorganizing from {source}...")
    try:
        dest = loader.prepare_brain_data(source)
        print(f"Successfully reorganized brain data to {dest}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    manual_reorganize()
