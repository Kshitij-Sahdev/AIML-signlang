import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
from model import check_gpu

app = Flask(__name__)

# Load Model
MODEL_PATH = 'sign_language_model.h5'
model = None

def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        check_gpu()
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Please run train.py first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    
    # Preprocess Image
    try:
        img = Image.open(file.stream).convert('L')  # Convert to Grayscale
        img = img.resize((64, 64))
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return jsonify({
            'class': int(predicted_class),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_trained_model()
    app.run(debug=True, port=5000)
