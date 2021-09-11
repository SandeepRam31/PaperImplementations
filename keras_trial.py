import tensorflow as tf 
from tensorflow.keras import layers, Model, Sequential 
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Input, Concatenate
import sys 
import numpy as np 
from tensorflow.keras.datasets import mnist

input_layer = Input(shape=(224, 224, 1))

x = MaxPooling2D(pool_size=3, strides=1, padding='same')(input_layer)
x = Conv2D(128, 1, strides=1, padding='same')(x)

y = Conv2D(64, 1, strides=1 , padding='same')(input_layer)
y = Conv2D(64, 5, strides=1, padding='same')(y)

z = Conv2D(64, 1, strides=1 , padding='same')(input_layer)
z = Conv2D(64, 3, strides=1, padding='same')(z)

w = Conv2D(128, 1, strides=1, padding='same')(input_layer)

x = Concatenate(axis=-1)([w, x, y, z])

model = Model(inputs = input_layer, outputs = x)

dot_img_file = 'tmp/model.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
