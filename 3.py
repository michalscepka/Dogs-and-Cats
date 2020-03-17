#------------------------------------------------------------------------------
#                           Automated model finding
#------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np
import time

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

#X = X/255.0
X = tf.keras.utils.normalize(X, axis=1)
y = np.array(y)

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}--conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs\{}'.format(NAME)) # 'tensorboard --logdir=logs'  shows tensorboard
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])
