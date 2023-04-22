import numpy as np
import tensorflow as tf
import cv2
import h5py
import os
# Load the model from the .h5 file
model_file = 'model.h5'
model_test = tf.keras.models.load_model(model_file)

# Load the image you want to predict on
user_input = input("Enter the path of your file: ")
try:
	os.path(user_input)
except:
	print("Please enter a valid path")
image_file = user_input
image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (200, 200))
image = np.expand_dims(image, axis=0)

# Make the prediction
prediction = model_test.predict(image)
print(str(prediction)+"\n")
if(prediction[0][0]<=0.1):
	res = "undistorted"
else:
	res = "blurred"
print(f"model predicted {prediction[0][0]*100} % probability that the image is {res}")
