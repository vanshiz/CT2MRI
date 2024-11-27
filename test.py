import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from PIL import Image
from InstanceNormalization import InstanceNormalization  # Replace with actual import

# Load model
custom_objects = {'InstanceNormalization': InstanceNormalization}
model = load_model('cycleGAN.h5', custom_objects=custom_objects)

# Load and preprocess image
image_path = "E:\\Minor2\\archive\\CT2MRI\\datasetCT\\split\\train\\no\\1 no.jpeg"
input_shape = (256, 256)  # Replace with model's required input size
image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=input_shape)
image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
input_data = np.expand_dims(image_array, axis=0)

# Run inference
predictions = model.predict(input_data)
# print(predictions)

# Post-process and visualize output
generated_image = np.squeeze(predictions)
generated_image = (generated_image * 255).astype(np.uint8)  # Rescale if needed

# Ensure 'generated' folder exists
output_folder = "generated"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the image in the 'generated' folder
output_image_path = os.path.join(output_folder, "generated_image.png")
output_image = Image.fromarray(generated_image)
output_image.save(output_image_path)

print(f"Image saved to {output_image_path}")