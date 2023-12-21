import io
import base64
import numpy as np
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

# Define CNN model using Keras
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Load model weights
model = create_model()
try:
    model.load_weights('./keras_model_weights.h5')
except Exception as e:
    print(f"Error loading model weights: {str(e)}")
    exit(1)

# Image preprocessing
def preprocess_image(image_data):
    try:
        # Decode base64 and convert to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')

        # Resize image to 28x28
        image = image.resize((28, 28))

        # Convert image to numpy array and normalize
        image_array = np.array(image) / 255.0

        # Reshape for model input
        image_array = image_array.reshape(1, 28, 28, 1)

        return image_array

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None


@app.route('/', methods=['GET'])
def welcome():
    return "Welcome to the Keras CNN prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json.get('image', None)
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        image = preprocess_image(image_data)
        if image is None:
            return jsonify({"error": "Failed to preprocess image"}), 400

        # Perform prediction
        predictions = model.predict(image)

        # Get the top two predictions and their corresponding probabilities
        top_two_indices = np.argsort(predictions[0])[-2:][::-1]  # Get indices of top 2 predictions
        top_prediction = int(top_two_indices[0])
        second_prediction = int(top_two_indices[1])

        # Get the probabilities for the top two predictions
        top_probability = predictions[0][top_prediction]
        second_probability = predictions[0][second_prediction]

        return jsonify({
            "prediction": top_prediction,
            "prediction_percentage": f"{top_probability * 100:.2f}%",
            "second_closest_prediction": second_prediction,
            "second_prediction_percentage": f"{second_probability * 100:.2f}%"
        })

    except Exception as e:
        print(f"Error predicting: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500




if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
