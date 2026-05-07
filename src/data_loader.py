import os
import shutil
import kagglehub
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, target_size=(224, 224), batch_size=32):
        self.target_size = target_size
        self.batch_size = batch_size
        self.dataset_root = 'dataset'

    def download_datasets(self):
        print("Downloading datasets from Kaggle...")
        # Brain Tumor MRI
        brain_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        # Chest X-Ray
        pneumonia_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        
        return brain_path, pneumonia_path

    def prepare_brain_data(self, source_path):
        """Reorganize brain tumor data into yes/no structure."""
        target_dir = os.path.join(self.dataset_root, 'brain_tumor')
        os.makedirs(target_dir, exist_ok=True)
        
        # Mapping: glioma, meningioma, pituitary -> yes | no_tumor -> no
        # The dataset has Training and Testing folders
        for split in ['Training', 'Testing']:
            split_path = os.path.join(source_path, split)
            for category in os.listdir(split_path):
                label = 'no' if 'no' in category.lower() else 'yes'
                dest_path = os.path.join(target_dir, label)
                os.makedirs(dest_path, exist_ok=True)
                
                cat_path = os.path.join(split_path, category)
                for img_name in os.listdir(cat_path):
                    shutil.copy(os.path.join(cat_path, img_name), os.path.join(dest_path, img_name))
        
        return target_dir

    def prepare_pneumonia_data(self, source_path):
        """Reorganize pneumonia data into standardized structure."""
        target_dir = os.path.join(self.dataset_root, 'pneumonia')
        os.makedirs(target_dir, exist_ok=True)
        
        # Pneumonia dataset usually has train/test/val
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(source_path, 'chest_xray', split)
            if not os.path.exists(split_path):
                continue
            for category in os.listdir(split_path):
                label = category.lower()
                dest_path = os.path.join(target_dir, label)
                os.makedirs(dest_path, exist_ok=True)
                
                cat_path = os.path.join(split_path, category)
                for img_name in os.listdir(cat_path):
                    shutil.copy(os.path.join(cat_path, img_name), os.path.join(dest_path, img_name))
        
        return target_dir

    def get_data_generators(self, data_dir):
        """Create train and test generators with augmentation."""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.15,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_gen = datagen.flow_from_directory(
            data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training'
        )

        test_gen = datagen.flow_from_directory(
            data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation'
        )

        return train_gen, test_gen

    def load_for_ml(self, data_dir):
        """Load images and labels into numpy arrays for ML feature extraction."""
        import cv2
        import numpy as np
        images = []
        labels = []
        classes = sorted(os.listdir(data_dir))
        
        for idx, cls in enumerate(classes):
            cls_path = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_path)[:500]: # Limit for demo speed if needed
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.target_size)
                    images.append(img)
                    labels.append(idx)
        
        X = np.array(images) / 255.0
        y = np.array(labels)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
