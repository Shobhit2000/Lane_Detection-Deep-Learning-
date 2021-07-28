import numpy as np
import matplotlib.pyplot as plt
import cv2
import os.path as ops
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import math

import architecture

x_train = []
y_train = []

# load data
for _, dirnames, files in os.walk("training/gt_image/"):
	i = 0
	for filenames in files:
		x_path = "training/gt_image/" + filenames
		y_path = "training/gt_binary_image/" + filenames

		x = cv2.imread(x_path)
		# x = cv2.resize(x, (256, 128))
		x = cv2.resize(x, (224, 224))
		x = x/255
		x_train.append(x)

		y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
		y = cv2.resize(y, (224, 224))
		y[y <= 120] = 0
		y[y > 120] = 255
		y = y/255
		y_train.append(y)

		i += 1
		#print(i)

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)
print(y_train.shape)
print("Data Loaded")

# Test Dataset
cv2.imshow('img', x_train[10])
cv2.waitKey(10)
cv2.destroyAllWindows()
cv2.imshow('img', x_train[10]*255)
cv2.waitKey(10)
cv2.destroyAllWindows()
cv2.imshow('img', y_train[10])
cv2.waitKey(10)
cv2.destroyAllWindows()
cv2.imshow('img', y_train[10]*255)
cv2.waitKey(10)
cv2.destroyAllWindows()

# Load Network Architecture
model = architecture.model_arch()
model.summary()
print("model summary")

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

# Train Network
history = model.fit(x_train, y_train, epochs= 15, batch_size=20, steps_per_epoch=180)

# Save Model
model.save('laneNet_model.h5')

# Display original, Prediction and Ground Truth images
img = cv2.imread('training/gt_image/0000.png')
img = cv2.resize(img, (224, 224))
cv2.imshow('original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = img/255
img = np.expand_dims(img, axis=0)
print(img.shape)
x = model.predict(img)

print(x.shape)
cv2.imshow('predicted', x[0]*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ground Truth Image
img = cv2.imread('training/gt_binary_image/0000.png')
img = cv2.resize(img, (224, 224))
cv2.imshow('Ground Truth', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
