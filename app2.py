import os
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained combined model
model = load_model('CombinedBrainTumorModel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Class mapping function
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Image at path {image_path} could not be loaded.")
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function for CT and MRI inputs
def getResult(ct_img_path, mri_img_path):
    try:
        ct_image = preprocess_image(ct_img_path)
        mri_image = preprocess_image(mri_img_path)
        
        # Perform prediction with both inputs
        result = model.predict([ct_image, mri_image])
        
        # Print the raw prediction result for debugging
        print(f"Raw prediction result: {result}")
        
        # Apply threshold (0.5) to classify as Tumor or No Tumor
        predicted_class = result[0][0]
        if predicted_class >= 0.5:
            return 1  # Tumor
        else:
            return 0  # No Tumor
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get uploaded files
        ct_file = request.files.get('ct_file')  # CT image file
        mri_file = request.files.get('mri_file')  # MRI image file

        if not ct_file or not mri_file:
            return "Error: Please upload both CT and MRI images."

        # Save the files to the uploads folder
        basepath = os.path.dirname(__file__)
        ct_path = os.path.join(basepath, 'uploads', secure_filename(ct_file.filename))
        mri_path = os.path.join(basepath, 'uploads', secure_filename(mri_file.filename))
        
        os.makedirs(os.path.dirname(ct_path), exist_ok=True)
        ct_file.save(ct_path)
        mri_file.save(mri_path)

        # Get the prediction
        value = getResult(ct_path, mri_path)
        if isinstance(value, str) and value.startswith("Error"):
            return value
        result = get_className(value)  # Map the result to the label
        return result  # Return the result as a string
    return None

if __name__ == '__main__':
    app.run(debug=True)
