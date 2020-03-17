#------------------------------------------------------------------------------
#                           Using saved/trained model
#------------------------------------------------------------------------------

import cv2
import tensorflow as tf

CATEGORIES = ["Dog","Cat"] # will use this to convert prediction num to string value

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # read in the image, convert to grayscale
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE)) # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # return the image with shaping that TF wants

model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('pictures/cat1.jpg')]) # always predict a list

print("Prediction: ", CATEGORIES[int(prediction[0][0])])
