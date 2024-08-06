from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from io import BytesIO

# Load the trained model
model_path = 'MRImodel.keras'  # Change this to your model's path
model = tf.keras.models.load_model(model_path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load and preprocess the image
    image = load_img(BytesIO(file.read()), target_size=(256, 256))  # Change target size if needed
    image_array = img_to_array(image)  # Convert image to array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    output = np.argmax(prediction[0])  # Get the index of the class with the highest probability

    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Use your model's labels
    prediction_text = 'Predicted Class: {}'.format(labels[output])

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
