import tensorflow as tf
from tensorflow import keras
from keras.layers import MaxPool2D
from keras.layers.pooling.base_pooling2d import Pooling2D
import numpy as np
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops.nn_ops import convert_padding
from tensorflow.python.ops.nn_ops import _get_sequence
from tensorflow.python.framework import ops
from keras.layers import Layer
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras import backend as K


class TiledConv2DLayer(Conv2D):
    """
    A customized Conv2D layer for implementing Tiled CNN
    """
    def convolution_op(self, inputs, kernel):
        # hard-coding step k = 2 here
        k = 2
        # untie weights by expanding kernel, equivelantly skipping pixel value
        # with step k
        rows = K.stack([kernel[0]] * k + [kernel[1]] + [kernel[2]] * k, axis=0)
        kernel_tiled = K.stack([rows[:, 0]] * k + [rows[:, 1]] + [rows[:, 2]] * k, axis=1)
        mean, var = tf.nn.moments(kernel_tiled, axes=[0, 1, 2], keepdims=True)
        return tf.nn.conv2d(
            inputs,
            (kernel_tiled - mean) / tf.sqrt(var + 1e-10),
            padding="VALID",
            strides=list(self.strides),
            name=self.__class__.__name__,
        )


def build_tcnn(init_scheme):
    model = Sequential()
    model.add(TiledConv2DLayer(input_shape=(64, 64, 1), filters=64, kernel_size=(5, 5), padding="valid", activation="relu", kernel_initializer=init_scheme))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu", kernel_initializer=init_scheme))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(TiledConv2DLayer(filters=128, kernel_size=(5, 5), padding="valid", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(TiledConv2DLayer(filters=256, kernel_size=(5, 5), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(TiledConv2DLayer(filters=512, kernel_size=(5, 5), padding="valid", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=6, activation="softmax"))
    return model

