import tensorflow as tf 
from tensorflow.keras import layers, Model, Sequential 
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Dense, Input, Concatenate
import sys 
import numpy as np 
from tensorflow.keras.datasets import cifar100


class ConvBlock(layers.Layer):
	def __init__(self, filters):
		super(ConvBlock, self).__init__()
		self.conv1 = Conv2D(filters, 3, strides=1, padding='same',)
		self.conv2 = Conv2D(filters, 3, strides=1, padding='same',)
		self.bn1 = BatchNormalization()
		self.bn2 = BatchNormalization()

	def call(self, input_tensor, training = False):
		x = self.conv1(input_tensor, training = training)
		x = self.bn1(x)
		x = self.conv2(x, training = training)
		x = self.bn2(x, training = training)
		return tf.nn.relu(x)

def UpBlock(filters, input_tensor, skip):
	x = Conv2DTranspose(filters, 2, strides = 2, padding='same')(input_tensor)
	x = Concatenate()([skip, x])
	return ConvBlock(filters)(x)

class UNet(Model):
	def __init__(self):
		super(UNet, self).__init__()
		self.block1 = ConvBlock(64)
		self.block2 = ConvBlock(128)
		self.block3 = ConvBlock(256)
		self.block4 = ConvBlock(512)
		self.block5 = ConvBlock(1024)

	def call(self, input_tensor):
		x_1 = self.block1(input_tensor)
		x_2 = MaxPooling2D()(x_1)
		x_2 = self.block2(x_2)
		x_3 = MaxPooling2D()(x_2)
		x_3 = self.block3(x_3)
		x_4 = MaxPooling2D()(x_3)
		x_4 = self.block4(x_4)
		x_5 = MaxPooling2D()(x_4)
		x_5 = self.block5(x_5)

		x = UpBlock(512, x_5, x_4)
		x = UpBlock(256, x, x_3)
		x = UpBlock(128, x, x_2)
		x = UpBlock(64, x, x_1)

		# return Conv2DTranspose(512, 2, strides = 2, padding='same')(x_5)
		return Conv2D(1, 1, padding='same', activation='sigmoid',)(x)

	def Model(self):
		x = Input((224, 224, 3))
		return Model(x, self.call(x))

unet_model = UNet().Model()
dot_img_file = 'tmp/UNet.jpg'
tf.keras.utils.plot_model(unet_model, to_file=dot_img_file, show_shapes=True)
