import tensorflow as tf 
from tensorflow.keras import layers, Model, Sequential 
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, MaxPool2D, DepthwiseConv2D, Conv1D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Dense, Input, Concatenate
import sys 
import numpy as np 

class CNNBlock(layers.Layer):
	def __init__(self, filters):
		super(CNNBlock, self).__init__()
		self.conv = DepthwiseConv2D(3, strides=1, padding='same', activation='relu',)
		self.bn1 = BatchNormalization()
		self.bn2 = BatchNormalization()
		self.depth_ = Conv2D(filters, 1, strides=1, padding='same', activation='relu',)

	def call(self, input_tensor):
		return self.bn2(self.depth_(self.bn1(self.conv(input_tensor))))

class ResBlock(layers.Layer):
	def __init__(self, output_channels = [32, 64, 128]):
		super(ResBlock, self).__init__()
		self.cnn1 = CNNBlock(output_channels[0])
		self.cnn2 = CNNBlock(output_channels[1])
		self.cnn3 = CNNBlock(output_channels[2])
		self.pooling = MaxPooling2D()
		self.identity_mapping = Conv2D(output_channels[1], 1, padding = 'same')

	def call(self, input_tensor, training = False):
		x = self.cnn1(input_tensor, training = training)
		x = self.cnn2(x, training = training)
		x = self.cnn3(
				x + self.identity_mapping(input_tensor), training=training
			)

		return x


class DepthConv(keras.Model):
	def __init__(self, num_classes = 10):
		super(DepthConv, self).__init__()
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
		
model = DepthConv()
dot_img_file = 'tmp/DepthConv.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
