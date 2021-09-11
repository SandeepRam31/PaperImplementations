import tensorflow as tf 
import numpy as np 
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow import keras
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1, 28*28)).astype('float32') / 255.0
x_test = np.reshape(x_test, (-1, 28*28)).astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

class MyRelu(layers.Layer):
	def __init(self):
		super(MyRelu, self).__init__()

	def call(self, input_tensor):
		return tf.math.maximum(0.0, input_tensor)

class Dense_Layer(layers.Layer):

	def __init__(self, units):
		super(Dense_Layer, self).__init__()
		self.units = units

	def call(self, input_tensor):
		return (tf.matmul(input_tensor, self.w) + self.b)

	def build(self, input_dims):

		self.w = self.add_weight(
			name = 'w',
			shape = (input_dims[-1], self.units),
			initializer = 'random_normal',
			trainable = True
			)

		self.b = self.add_weight(
				name = 'b',
				shape = (self.units, ),
				initializer = 'ones',
				trainable = True
			)


class MyModel(keras.Model):
	def __init__(self, num_classes = 10):
		super(MyModel, self).__init__()
		self.dense1 = Dense_Layer(64)
		self.dense2 = Dense_Layer(num_classes)
		self.activation = MyRelu()

	def call(self, input_tensor):
		x = self.dense1(input_tensor)
		x = self.activation(x)
		x = self.dense2(x)
		return x

	def Model(self):
		x = Input(shape = (784))
		return Model(inputs = x, outputs = self.call(x))

simple = MyModel()
simple.compile(
	loss = 'categorical_crossentropy',
	metrics = 'accuracy',
	optimizer = 'adam'
	)

simple.fit(x_train, y_train, epochs = 1)

#############################
# if you use Build function in your custom layer, the second argument
# in build(self, second_argument) is the shape of the tensor it recieves
# if you use Call function in your custom layer, the second argument in 
# the call(self, second_argument) is the tensor itself that it recieces
# self.add_weight essentially just creates a random matrix of a given shape
# while creating custom layers always name each weight matrix, or else model cant be saved

