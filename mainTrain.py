import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Activation,Dropout,Flatten,Dense 


image_directory = 'brain_tumor_dataset/'

dataset = []
label = []

# List images for 'no' and 'yes' categories
no_tumor_images = os.listdir(image_directory + 'no/')
yes_tumor_images = os.listdir(image_directory + 'yes/')

# Process 'no' images
for i, image_name in enumerate(no_tumor_images):
    if image_name.endswith('.jpg'):  # Check file extension
        image_path = os.path.join(image_directory, 'no', image_name)
        image = cv2.imread(image_path)
        if image is not None:  # Check if the image was loaded
            image = Image.fromarray(image, 'RGB')
            image = image.resize((64, 64))
            dataset.append(np.array(image))
            label.append(0)
        else:
            print(f"Failed to load image: {image_path}")

# Process 'yes' images
for i, image_name in enumerate(yes_tumor_images):  
    if image_name.endswith('.jpg'):
        image_path = os.path.join(image_directory, 'yes', image_name)
        image = cv2.imread(image_path)
        if image is not None: 
            image = Image.fromarray(image, 'RGB')
            image = image.resize((64, 64))
            dataset.append(np.array(image))
            label.append(1)
        else:
            print(f"Failed to load image: {image_path}")

print(f"Dataset size: {len(dataset)}")
print(f"Label size: {len(label)}")


dataset=np.array(dataset)
label=np.array(label)
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)


x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)
INPUT_SIZE=64

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=50,validation_data=(x_test,y_test),shuffle=False)

model.save('BrainTumor10Epochs.h5')
