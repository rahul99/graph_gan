'''
Implement U-Net_like Architecture here
Input:
[1] Input image of dimension [row x col x channel]
[2] Hyper parameters while instantiating this class

Returns:
Output of the generator: an image of the same size as the input image (hxhx3) [3 (rgb) channels]
'''

import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers.merge import Concatenate
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import initializers
from keras import regularizers

class UNet:

	def __init__(self, my_img_rows, my_img_cols, my_channel_dim, my_out_channel_dim, my_kernel_size, my_pool_size):
		self.img_rows = my_img_rows
		self.img_cols = my_img_cols
		self.channel_dim = my_channel_dim
		self.out_channel_dim = my_out_channel_dim
		self.kernel_size = my_kernel_size
		self.pool_size = my_pool_size
		self.model = None


	def get_small_model(self):
		set_kernel_initializer = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
		set_bias_initializer = initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
		model = Sequential()
		model.add(Conv2D(30, (self.kernel_size, self.kernel_size), padding='same', input_shape=(self.img_rows, self.img_cols, self.channel_dim),
			kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer))
		model.add(Activation('tanh'))
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))

		model.add(UpSampling2D(size=(self.pool_size, self.pool_size)))
		model.add(Conv2D(3, (self.kernel_size, self.kernel_size), padding='same',
			kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer))
		model.add(Activation('tanh'))
		return(model)


	def get_model(self):
		my_input = Input((self.img_rows, self.img_cols, self.channel_dim))

		set_kernel_initializer = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
		set_bias_initializer = initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

		set_kernel_regularizer = regularizers.l1_l2(1e-6)
		set_bias_regularizer = regularizers.l1_l2(1e-6)
		set_activity_regularizer = regularizers.l1_l2(1e-20)

		conv1 = Conv2D(64, (self.kernel_size, self.kernel_size), padding='same', input_shape=(self.img_rows, self.img_cols, self.channel_dim), kernel_initializer=set_kernel_initializer, activation='tanh',
			bias_initializer=set_bias_initializer, kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(my_input)
		#pool1 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv1)
		pool1 = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))(conv1)		

		conv2 = Conv2D(128, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh', bias_initializer=set_bias_initializer,
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(pool1)
		#pool2 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv2)
		pool2 = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))(conv2)

		conv3 = Conv2D(256, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh', bias_initializer=set_bias_initializer,
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(pool2)
		#pool3 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv3)
		pool3 = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))(conv3)		

		conv4 = Conv2D(512, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh', bias_initializer=set_bias_initializer,
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(pool3)

		up1 = Conv2D(256, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh', bias_initializer=set_bias_initializer,
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(UpSampling2D(size=(self.pool_size, self.pool_size))(conv4))
		concat1 = Concatenate(axis=3)([up1, conv3])

		up2 = Conv2D(128, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh', bias_initializer=set_bias_initializer,
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(UpSampling2D(size=(self.pool_size, self.pool_size))(concat1))
		concat2 = Concatenate(axis=3)([up2, conv2])

		up3 = Conv2D(64, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh', bias_initializer=set_bias_initializer,
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(UpSampling2D(size=(self.pool_size, self.pool_size))(concat2))
		concat3 = Concatenate(axis=3)([up3, conv1])

		#my_output = Conv2D(self.out_channel_dim, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh',
		#	bias_initializer=set_bias_initializer, kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(concat3)

		my_output = Conv2D(self.out_channel_dim, (1,1), padding='same', kernel_initializer=set_kernel_initializer, activation='tanh',
			bias_initializer=set_bias_initializer, kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(concat3)

		self.model = Model(inputs=my_input, outputs=my_output)

		return(self.model)