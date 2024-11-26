import cv2
import os
from PIL import Image
import numpy as np
from keras.utils import normalize
from keras.models import Model
from keras.layers import (
    Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Input, Concatenate
)

input_size = 64

# Function to process images from a specific directory
def load_images_from_folder(folder_path):
    dataset = []
    labels = []
    for label, subfolder in enumerate(['no', 'yes']):  # Assuming subfolders 'no' and 'yes'
        path = os.path.join(folder_path, subfolder)
        for image_name in os.listdir(path):
            if image_name.endswith('.jpg'):  # Only process .jpg files
                image_path = os.path.join(path, image_name)
                image = cv2.imread(image_path)
                if image is not None:  # Ensure the image is loaded correctly
                    image = Image.fromarray(image, 'RGB')
                    image = image.resize((input_size, input_size))
                    dataset.append(np.array(image))
                    labels.append(label)
    return np.array(dataset), np.array(labels)

# Load train, test, and validation datasets for CT and MRI
def load_split_data(base_directory):
    train_images, train_labels = load_images_from_folder(os.path.join(base_directory, 'train'))
    test_images, test_labels = load_images_from_folder(os.path.join(base_directory, 'test'))
    val_images, val_labels = load_images_from_folder(os.path.join(base_directory, 'val'))
    return train_images, train_labels, test_images, test_labels, val_images, val_labels

# Load MRI and CT datasets
ct_base_dir = 'datasetCT\split'
mri_base_dir = 'datasetMRI\mri\split'

x_ct_train, y_ct_train, x_ct_test, y_ct_test, x_ct_val, y_ct_val = load_split_data(ct_base_dir)
x_mri_train, y_mri_train, x_mri_test, y_mri_test, x_mri_val, y_mri_val = load_split_data(mri_base_dir)

# Ensure labels match across modalities
assert (y_ct_train == y_mri_train).all(), "CT and MRI training labels do not match!"
assert (y_ct_test == y_mri_test).all(), "CT and MRI testing labels do not match!"
assert (y_ct_val == y_mri_val).all(), "CT and MRI validation labels do not match!"

# Normalize the images
x_ct_train = normalize(x_ct_train, axis=1)
x_ct_test = normalize(x_ct_test, axis=1)
x_ct_val = normalize(x_ct_val, axis=1)

x_mri_train = normalize(x_mri_train, axis=1)
x_mri_test = normalize(x_mri_test, axis=1)
x_mri_val = normalize(x_mri_val, axis=1)

# Create CT and MRI sub-models
def create_submodel(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), kernel_initializer='he_uniform')(input_layer)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), kernel_initializer='he_uniform')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Model(inputs=input_layer, outputs=x)

# Define input shape
input_shape = (input_size, input_size, 3)

# Create models for CT and MRI
ct_model = create_submodel(input_shape)
mri_model = create_submodel(input_shape)

# Combine models
combined = Concatenate()([ct_model.output, mri_model.output])
x = Dense(128, activation='relu')(combined)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

# Define the final model
merged_model = Model(inputs=[ct_model.input, mri_model.input], outputs=output)

# Compile the model
merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using the pre-split dataset
merged_model.fit(
    [x_ct_train, x_mri_train],
    y_ct_train,
    batch_size=16,
    epochs=50,
    validation_data=([x_ct_val, x_mri_val], y_ct_val),
    shuffle=True
)

# Save the model
merged_model.save('CombinedBrainTumorModel.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = merged_model.evaluate([x_ct_test, x_mri_test], y_ct_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
