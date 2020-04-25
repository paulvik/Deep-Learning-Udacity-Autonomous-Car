from numpy.random import seed
seed(1)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Cropping2D, Lambda
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np


def resizing(image):
    import tensorflow as tf
    new_img = tf.image.resize_images(image, size=[66,200])
    return new_img



def neural_net():
    act = 'relu'
    init = 'he_normal'

    model = Sequential()
    
    model.add(Lambda(resizing))

  # Convolutional layers
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), padding='valid', activation=act, kernel_initializer=init))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), padding='valid', activation=act, kernel_initializer=init))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), padding='valid', activation=act, kernel_initializer=init))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', activation=act, kernel_initializer=init))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', activation=act, kernel_initializer=init))
    model.add(Flatten())

  # Fully-connected layers
    model.add(Dense(100, activation=act, kernel_initializer=init))
    model.add(Dense(50, activation=act, kernel_initializer=init))
    model.add(Dense(10, activation=act, kernel_initializer=init))
    model.add(Dense(1, kernel_initializer=init))
    return model




