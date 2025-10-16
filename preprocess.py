import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def getDataset(img_size=(128, 128), batch_size=32):
    # Téléchargement via KaggleHub
    path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
    print("Path to dataset files:", path)

    train_dir = os.path.join(path, "Face Mask Dataset", "Train")
    val_dir = os.path.join(path, "Face Mask Dataset", "Validation")

    # Data augmentation pour le train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_gen, val_gen