#------------------------------------------------------------------------------
#                      Model setup and TensorBoard
#------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time

NAME = "Cats-vs-dogs-CNN-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME)) # 'tensorboard --logdir=logs'  shows tensorboard

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

#X = X/255.0
X = tf.keras.utils.normalize(X, axis=1)
y = np.array(y)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors

#model.add(Dense(64))
#model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=20, validation_split=0.2, callbacks=[tensorboard])
