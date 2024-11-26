import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the combined model
model = load_model('CombinedBrainTumorModel.h5')
print("Model loaded successfully!")

# Load and preprocess an image for a specific modality
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    img = Image.fromarray(image, 'RGB')
    img = img.resize((64, 64))  # Resize to match model's input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Paths to the CT and MRI images for the same patient
ct_image_path = 'E:\\Minor2\\archive\\datasetCT\\split\\test\\no\\3 no.jpg'  # Replace with actual path
mri_image_path = 'E:\\Minor2\\archive\\datasetMRI\\mri\\split\\test\\no\\mri_3 no.jpg'  # Replace with actual path

# Preprocess both images
ct_image = preprocess_image(ct_image_path)
mri_image = preprocess_image(mri_image_path)

# Make a prediction
result = model.predict([ct_image, mri_image])

# Interpret the prediction
predicted_class = result[0][0]
if predicted_class >= 0.5:
    print("Prediction: Tumor")
else:
    print("Prediction: Not Tumor")
