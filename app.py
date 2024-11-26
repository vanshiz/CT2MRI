import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Class mapping function
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Prediction function with thresholding for sigmoid output
def getResult(img):
    image = cv2.imread(img)
    if image is None:
        return "Error: Image not loaded correctly"

    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))  # Resize the image to match the model's input size
    image = np.array(image)

    # Normalize the image (assuming the model expects inputs in range [0, 1])
    image = image / 255.0  # Normalize pixel values to [0, 1]

    # Check the shape of the image
    print(f"Image shape after resizing and normalization: {image.shape}")

    input_img = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    result = model.predict(input_img)
    
    # Print the raw prediction result for debugging
    print(f"Raw prediction result: {result}")
    
    # If the model uses sigmoid (binary classification), result will be a probability
    predicted_class = result[0][0]  # Single output for sigmoid (probability)
    
    # Apply threshold (0.5) to classify as Tumor or Not Tumor
    if predicted_class >= 0.5:
        return 1  # Tumor
    else:
        return 0  # No Tumor

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to the uploads folder
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Get the prediction
        value = getResult(file_path)
        result = get_className(value)  # Map the result to the label
        return result  # Return the result as a string (either "No Brain Tumor" or "Yes Brain Tumor")
    return None

if __name__ == '__main__':
    app.run(debug=True)
