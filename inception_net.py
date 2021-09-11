import tensorflow as tf 
from tensorflow.keras import layers, Model, Sequential 
from tensorflow import keras
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Input, Concatenate
import sys 
import numpy as np 
from tensorflow.keras.datasets import cifar100

# import dataset here 

# x_train = np.reshape(x_train, (-1, 224, 224, 3)).astype('float32') / 255.0
# x_test = np.reshape(x_test, (-1, 224, 224, 3)).astype('float32') / 255.0

# y_train = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)

# inception block input-> 3x3 Maxpooling , 1x1 5x5, 1x1 3x3, 1x1

class ConvBlock(layers.Layer):
	def __init__(self, output_channels, kernals, strides, padding):
		super(ConvBlock, self).__init__()
		self.conv_1 = Conv2D(output_channels, kernals, strides=strides, padding=padding)
		self.bn = BatchNormalization()

	def call(self, inputs, training = False):
		x = self.conv_1(inputs, training = training)
		x = self.bn(x, training = training)
		return tf.nn.relu(x)

class InceptionBlock(layers.Layer):
	def __init__(self, conv1, conv3_reduce, conv3, conv5_reduce, conv5, pool_projection):
		super(InceptionBlock, self).__init__()
		self.conv_1 = ConvBlock(conv1, 1, padding='same', strides=1)
		self.conv_2 = ConvBlock(conv3, 3, padding='same', strides=1)
		self.conv_3 = ConvBlock(conv5, 5, padding='same', strides=1)
		self.conv_4 = ConvBlock(pool_projection, 1, padding='same', strides=1)

		self.identity_1 = ConvBlock(conv3_reduce, 1, padding='same', strides=1)
		self.identity_2 = ConvBlock(conv5_reduce, 1, padding='same', strides=1)

		self.pool = MaxPooling2D(pool_size=3, padding = 'same', strides=1)
		self.concat = Concatenate(axis = -1)

	def call(self, input_tensor, training = False):
		branch_1 = self.conv_1(input_tensor)

		id_branch_2 = self.identity_1(input_tensor)
		branch_2 = self.conv_2(id_branch_2)

		id_branch_3 = self.identity_2(input_tensor)
		branch_3 = self.conv_3(id_branch_3)

		branch_4 = MaxPooling2D(pool_size=3, padding = 'same', strides=1)(input_tensor)
		branch_4 = self.conv_4(branch_4)

		return self.pool(self.concat([branch_1, branch_2, branch_3, branch_4]))

	def Model(self):
		x = Input((224, 224, 3))
		return Model(x, self.call(x))

class InceptionModel(keras.Model):

	def __init__(self):
		super(InceptionModel, self).__init__()
		self.conv1 = ConvBlock(output_channels = 32, kernals = 3, padding='same', strides=1)
		self.pool = MaxPooling2D(3, 2)
		self.conv2 = ConvBlock(output_channels = 32, kernals = 3, padding='same', strides=1)

		self.block3a = InceptionBlock(64, 96, 128, 16, 32, 32)
		self.block3b = InceptionBlock(128, 128, 192, 32, 96, 64)

		self.block4a = InceptionBlock(192, 96, 208, 16, 48, 64)
		self.block4b = InceptionBlock(160, 112, 224, 24, 64, 64)
		self.block4c = InceptionBlock(128, 128, 256, 24, 64, 64)
		self.block4d = InceptionBlock(112, 144, 288, 32, 64, 64)
		self.block4e = InceptionBlock(256, 160, 320, 32, 128, 128)

		self.block5a = InceptionBlock(256, 160, 320, 32, 128, 128)
		self.block5b = InceptionBlock(384, 192, 384, 48, 128, 128)

		self.avgpool = GlobalAveragePooling2D()
		self.drop = Dropout(0.4)
		self.final_layer = Dense(1000, activation='softmax')

	def call(self, input_tensor):

		x = self.conv1(input_tensor)
		x = self.conv2(x)
		x = MaxPooling2D()(x)

		x = self.block3a(x)
		x = self.block3b(x)
		x = MaxPooling2D()(x)

		x = self.block4a(x)
		x = self.block4b(x)
		x = self.block4c(x)
		x = self.block4d(x)
		x = self.block4e(x)
		x = MaxPooling2D()(x)

		x = self.block5a(x)
		x = self.block5b(x)
		x = self.avgpool(x)
		x = self.drop(x)
		output_layer = self.final_layer(x)

		return output_layer

	def model(self):
		input_layer = Input(shape=(1024, 1024, 3))
		return Model(inputs = input_layer, outputs = self.call(input_layer))

model = InceptionModel().model()

dot_img_file = 'tmp/inceptionModel.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model = InceptionBlock(64, 96, 128, 16, 32, 32).Model()

dot_img_file = 'tmp/InceptionBlock.jpg'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)