# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 00:52:33 2022

@author: PRAMILA
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#Load dataset
import pathlib
data_dir = pathlib.Path("C:/Users/PRAMILA/.spyder-py3/ibm/data/train")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)



#Read monuments images from disk into numpy array using opencv
monuments_images_dict = {
    'fire': list(data_dir.glob('fire/*')),
    'nofire': list(data_dir.glob('nofire/*'))
}

monuments_labels_dict = {
    'fire': 0,
    'nofire': 1
}

monuments_dict =['fire','nofire']

#Train test split
X, y = [], []

for monument_name, images in monuments_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(monuments_labels_dict[monument_name])

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

num_classes = 2

model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(X_train_scaled, y_train, epochs=50)             

model.save('model.h5')



