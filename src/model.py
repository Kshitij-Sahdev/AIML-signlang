import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(64, 64, 1), num_classes=6):
    """
    Creates a simple CNN model as described in the report.
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Detected: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU detected. Training will run on CPU.")
