import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class Config:
    img_size=(256,256)
    batch_size=4
    epochs=10
    learning_rate=1e-4


class Dataset(keras.utils.Sequence):
    def __init__(self, img_list, mask_list, transform, batch_size=4):
        self.img_list = img_list  # Full path
        self.mask_list = mask_list
        self.transform = transform
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.img_list) / self.batch_size))
        
    def __getitem__(self, index):
        batch_img_paths = self.img_list[index * self.batch_size : (index + 1) * self.batch_size]
        batch_mask_paths = self.mask_list[index * self.batch_size : (index + 1) * self.batch_size]

        images, masks = [], []
        for img_path, mask_path in zip(batch_img_paths, batch_mask_paths):
            img = Image.open(img_path).convert("RGB")  
            mask = Image.open(mask_path).convert("L")

            mask = np.array(mask) / 255.0
            mask[mask > 0.5] = 1

            augmented = self.transform(image=np.array(img), mask=mask)
            images.append(augmented["image"])
            masks.append(np.expand_dims(augmented["mask"], axis=-1))

        return np.array(images), np.array(masks)
