#------------------------------------------------------------------------------
#                           Using saved/trained model
#------------------------------------------------------------------------------

import cv2
import tensorflow as tf
import os   # directories and paths in os
import matplotlib.pyplot as plt     # show the image

CATEGORIES = ["Dog","Cat"] # will use this to convert prediction num to string value
DATADIR = "C:/Users/Michal/Dropbox/Neural network/dogs_and_cats/pictures"
model = tf.keras.models.load_model("3x64x0-CNN-75.model")
IMG_SIZE = 75

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # return the image with shaping that TF wants

def predict_images(datadir):
    path = os.path.join(datadir)
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        try:
            prediction = model.predict([prepare(os.path.join(path, img))]) # always predict a list
            print("Prediction for {}: ".format(img), CATEGORIES[int(prediction[0][0])])

            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # read in the image, convert to grayscale
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize image to match model's expected sizing
            plt.imshow(new_array, cmap="gray")
            plt.show()
        except Exception as e:
            print(e)

predict_images(DATADIR)
