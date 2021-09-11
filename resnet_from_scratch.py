import tensorflow as tf 
import numpy as np 
import os
import sys
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Add, BatchNormalization, Dropout, MaxPool2D, GlobalAveragePooling2D, Flatten
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = np.reshape(x_train, (-1, 28, 28, 1)).astype('float32') / 255.0
# x_test = np.reshape(x_test, (-1, 28, 28, 1)).astype('float32') / 255.0

# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

class CNNBlock(layers.Layer):
	def __init__(self, output_channels, kernals = 3):
		super(CNNBlock, self).__init__()
		self.cnn = Conv2D(output_channels, kernals, padding = 'same')
		self.bn = BatchNormalization()

	def call(self, input_tensor, training = True):
		x = self.cnn(input_tensor)
		x = self.bn(x, training = training)
		return tf.nn.relu(x)


class ResBlock(layers.Layer):
	def __init__(self, output_channels = [32, 64, 128]):
		super(ResBlock, self).__init__()
		self.cnn1 = CNNBlock(output_channels[0])
		self.cnn2 = CNNBlock(output_channels[1])
		self.cnn3 = CNNBlock(output_channels[2])
		self.pooling = MaxPool2D()
		self.identity_mapping = Conv2D(output_channels[1], 1, padding = 'same')

	def call(self, input_tensor, training = False):
		x = self.cnn1(input_tensor)
		x = self.cnn2(x)
		skip = self.identity_mapping(input_tensor)
		y = Add()([x, skip])
		return y

	def model(self):
		input_layer = Input(shape=(1024, 1024, 3))
		return Model(inputs = input_layer, outputs = self.call(input_layer))

class ResNet_Like(keras.Model):
	def __init__(self, num_classes = 10):
		super(ResNet_Like, self).__init__()
		self.block1 = ResBlock([32,32,64])
		self.block2 = ResBlock([32,32,64])
		self.block3 = ResBlock([64, 64, 128])
		self.block4 = ResBlock([64, 64, 128])
		self.block5 = ResBlock([64, 64, 128])
		self.block6 = ResBlock([64, 64, 128])
		self.block7 = ResBlock([128, 256, 512])
		self.block8 = ResBlock([128, 256, 512])
		self.gapool = GlobalAveragePooling2D()
		self.pool = MaxPool2D()
		self.classifier = layers.Dense(num_classes, activation = 'softmax')

	def call(self, input_tensor, training = False):
		y = self.block1(input_tensor)
		y = self.block2(y)
		y = self.block3(y)
		y = self.pool(y)
		y = self.block4(y)
		y = self.block5(y)
		y = self.pool(y)
		y = self.block6(y)
		y = self.block7(y)
		y = self.pool(y)
		y = self.block8(y)
		y = self.gapool(y)
		return self.classifier(y)

	def Model(self):
		input_layer = Input(shape = (224, 224, 3))
		return keras.Model(inputs = input_layer, outputs = self.call(input_layer))
		
model = ResNet_Like().Model()
dot_img_file = 'tmp/ResNet.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model = ResBlock([32,32,64]).model()
dot_img_file = 'tmp/ResBlock.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
