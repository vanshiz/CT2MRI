import cv2
from keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('BrainTumor10Epochs.h5')
print("Model loaded successfully!")


# image = cv2.imread('E:\\Minor2\\archive\\yes\\Y38.jpg')
image = cv2.imread('E:\\Minor2\\archive\\no\\9 no.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))  
img = np.array(img)


img = img / 255.0  
input_img = np.expand_dims(img, axis=0)

result = model.predict(input_img)

predicted_class = result[0][0]  
if predicted_class >= 0.5:
    print("Tumor")
else:
    print("Not Tumor")
