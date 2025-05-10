from flask import Flask, jsonify, request, send_from_directory
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tempfile
import imghdr
import json
import logging
import time
from flask_cors import CORS
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("models", "ai_crops_model.keras")
CLASS_INDEX_PATH = os.path.join("models", "class_indices.json")

try:
    model = load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

try:
    with open(CLASS_INDEX_PATH, 'r') as f:
        class_indices = json.load(f)
    CLASS_NAMES = [None] * len(class_indices)
    for class_name, index in class_indices.items():
        CLASS_NAMES[index] = class_name
    logger.info(f"Loaded {len(CLASS_NAMES)} class names successfully")
except Exception as e:
    logger.error(f"Failed to load class indices: {e}")
    raise

IMG_SIZE = (224, 224)

def preprocess_image(image_file):
    """Preprocess uploaded image file into model input."""
    if imghdr.what(image_file) not in ['jpeg', 'png']:
        raise ValueError('Unsupported image format. Use JPEG or PNG.')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image_file.save(temp_file.name)
        img = load_img(temp_file.name, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0  # match training
        img_array = np.expand_dims(img_array, axis=0)
    os.unlink(temp_file.name)
    return img_array

@app.route('/health', methods=['GET'])
def health_check():
    """Health check route."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict plant disease from uploaded image."""
    start_time = time.time()
    logger.info('Prediction request received')

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        processed_image = preprocess_image(image_file)
        prediction = model.predict(processed_image, verbose=0)
        predicted_class_index = np.argmax(prediction)
        confidence = prediction[0][predicted_class_index]
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        logger.info(f"Predicted {predicted_class_name} with confidence {confidence:.4f} in {time.time() - start_time:.2f}s")
        return jsonify({
            'class': predicted_class_name,
            'confidence': round(float(confidence), 4)
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
