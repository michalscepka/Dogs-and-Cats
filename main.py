#------------------------------------------------------------------------------
#                           Using saved/trained model
#------------------------------------------------------------------------------

import cv2
import tensorflow as tf
import os   # directories and paths in os

CATEGORIES = ["Dog","Cat"] # will use this to convert prediction num to string value
DATADIR = "C:/Users/Michal/Dropbox/Neural network/dogs_and_cats/pictures"
model = tf.keras.models.load_model("3x32x1-CNN.model")

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # read in the image, convert to grayscale
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE)) # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # return the image with shaping that TF wants

def predict_images(datadir):
    path = os.path.join(datadir)
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        try:
            prediction = model.predict([prepare(os.path.join(path, img))]) # always predict a list
            print("Prediction for {}: ".format(img), CATEGORIES[int(prediction[0][0])])
        except Exception as e:
            pass

predict_images(DATADIR)
