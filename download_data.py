from src.data_loader import DataLoader
import os

def main():
    loader = DataLoader()
    print("--- Initializing Dataset Download ---")
    
    try:
        brain_raw, pneumonia_raw = loader.download_datasets()
        
        print(f"Brain MRI raw path: {brain_raw}")
        print(f"Pneumonia raw path: {pneumonia_raw}")
        
        print("Reorganizing Brain MRI dataset...")
        brain_dir = loader.prepare_brain_data(brain_raw)
        print(f"Brain MRI ready at: {brain_dir}")
        
        print("Reorganizing Pneumonia dataset...")
        pneumonia_dir = loader.prepare_pneumonia_data(pneumonia_raw)
        print(f"Pneumonia ready at: {pneumonia_dir}")
        
        print("\nAll datasets downloaded and prepared successfully!")
        print("You can now run 'python main.py' to start the experiments.")
        
    except Exception as e:
        print(f"Error during download: {e}")
        print("\nNote: If you see a Kaggle credential error, please ensure you have 'kaggle.json' in your ~/.kaggle/ folder.")

if __name__ == "__main__":
    main()
