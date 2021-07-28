import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import math

def model_arch():
	model = ResNet50(weights="imagenet", include_top=False,
	input_tensor=layers.Input(shape=(224, 224, 3)))

	x = model.output
	 
	# Adding a Global Average Pooling layer
	x = GlobalAveragePooling2D()(x)

	x = layers.Conv2DTranspose(512, kernel_size=(3, 3), strides=2, activation="relu", padding="same")(model.layers[-1].output)
	x = layers.Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same")(x)

	x = layers.Conv2DTranspose(256, kernel_size=(3, 3), strides=2, activation="relu", padding="same")(x)
	x = layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")(x)
	x = layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)

	x = layers.Conv2DTranspose(64, kernel_size=(3, 3), strides = 2, activation="relu", padding="same")(x)
	x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)

	x = layers.Conv2DTranspose(64, kernel_size=(3, 3), strides = 2, activation="relu", padding="same")(x)
	x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)

	x = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides = 2, activation="relu", padding="same")(x)
	x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(x)

	outputs = layers.Conv2D(1, kernel_size=(3, 3), activation="sigmoid", padding="same")(x)

	model = tf.keras.Model(inputs=model.inputs, outputs=outputs, name="Lane_seg_binary")
	model.summary()

	for layer in model.layers:
		layer.trainable = True
	
	return model
