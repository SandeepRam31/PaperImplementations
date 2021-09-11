import tensorflow as tf 
from tensorflow.keras import layers, Model, Sequential 
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, MaxPool2D, DepthwiseConv2D, Conv1D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Dense, Input, Concatenate, Multiply
import sys 
import numpy as np 

class CNNBlock(layers.Layer):
	def __init__(self, filters):
		super(CNNBlock, self).__init__()
		self.conv = DepthwiseConv2D(3, strides=1, padding='same', activation='relu',)
		self.bn1 = BatchNormalization()
		self.bn2 = BatchNormalization()
		self.depth_ = Conv2D(filters, 1, strides=1, padding='same', activation='relu',)

	def call(self, input_tensor, training = False):
		return self.bn2(
				self.depth_(
					self.bn1(
						self.conv(input_tensor), training = training
						)
					), training = training
				)
	def model(self):
		input_layer = Input(shape=(1024, 1024, 3))
		return Model(inputs = input_layer, outputs = self.call(input_layer))

class Excite(layers.Layer):
	def __init__(self, reductive_ratio):
		super(Excite, self).__init__()
		self.r = reductive_ratio

	def call(self, input_tensor):
		shape = input_tensor.shape[-1]
		x = GlobalAveragePooling2D()(input_tensor)
		x = Dense(int(shape*self.r), activation='relu', use_bias='False')(x)
		x = Dense(shape, activation='sigmoid', use_bias='False')(x)
		return Multiply()([x, input_tensor])

	def model(self):
		input_layer = Input(shape=(512, 512, 64))
		return Model(inputs = input_layer, outputs = self.call(input_layer))

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

		return Excite(reductive_ratio = 0.5)(x)

class BetterNet(Model):
	def __init__(self):
		super(BetterNet, self).__init__()

		self.model = Sequential(layers=[
			ResBlock([256, 256, 128]),
			Conv2D(128, 3, padding='same', activation='relu'),
			ResBlock([128, 128, 64]),
			Conv2D(64, 3, padding='same', activation='relu'),
			ResBlock([64, 64, 32]),
			Conv2D(64, 3, padding='same', activation='relu'),
			ResBlock([64, 64, 32]),
			Conv2D(64, 3, padding='same', activation='relu'),
			ResBlock([64, 64, 32]),
			Conv2D(64, 3, padding='same', activation='relu'),
			ResBlock([64, 64, 32]),
			Conv2D(64, 3, padding='same', activation='relu'),
			ResBlock([64, 64, 32]),
			Conv2D(64, 3, padding='same', activation='relu'),
			ResBlock([32, 32, 16]),
			GlobalAveragePooling2D(),
			Dense(4, activation='softmax',)
			])

	def call(self, input_tensor):
		return self.model(input_tensor)

	def Model(self):
		x = Input(shape=(224, 224, 3))
		return Model(inputs=x, outputs=self.call(x))

model = BetterNet().Model()
dot_img_file = 'tmp/squeezeModel.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model = CNNBlock(128).model()
dot_img_file = 'tmp/squeezeDepth.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model = Excite(0.5).model()
dot_img_file = 'tmp/squeezeExcite.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
