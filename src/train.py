import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model, check_gpu

# Configuration
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = 'dataset'  # User needs to put data here
MODEL_SAVE_PATH = 'sign_language_model.h5'

def train():
    check_gpu()
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset not found at '{DATASET_DIR}'. Please download the dataset and extract it there.")
        print("Structure should be: dataset/train/0/image.jpg, dataset/test/0/image.jpg, etc.")
        return

    # Data Preprocessing & Augmentation
    # Report says: Grayscale, 64x64, scaled 0-1
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )

    print("Loading Validation Data...")
    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )

    # Create and Compile Model
    model = create_model(input_shape=(64, 64, 1), num_classes=train_generator.num_classes)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\n🚀 Starting Training on RTX (if available)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
