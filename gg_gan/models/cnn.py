'''
Implement CNN network here
Input:
[1] Input image of dimension [row x col x channel]
[2] Hyper parameters while instantiating this class
Returns:
Probability estimate: one for a true sample and zeor for a fake sample
'''

import sys
sys.path.append('./')
sys.path.append('../')

import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, merge, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras import initializers
from keras import regularizers

class CNN:
	def __init__(self, my_img_rows, my_img_cols, my_channel_dim, my_kernel_size, my_pool_size, my_l2_regularizer, my_dropout, my_labels):
		self.img_rows = my_img_rows
		self.img_cols = my_img_cols
		self.channel_dim = my_channel_dim
		self.kernel_size = my_kernel_size
		self.pool_size = my_pool_size		
		self.l2_regularizer = my_l2_regularizer #0.001
		self.dropout = my_dropout
		self.labels = my_labels
		self.model = None


	def get_small_model(self):
		set_kernel_initializer = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
		set_bias_initializer = initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)		

		model = Sequential()

		model.add(Conv2D(32, (self.kernel_size, self.kernel_size), padding='same', input_shape=(self.img_rows, self.img_cols, self.channel_dim), 
			kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer))
		model.add(Activation('tanh'))
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))

		model.add(Conv2D(64, (self.kernel_size, self.kernel_size), padding='same',
			kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer))
		model.add(Activation('tanh'))
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))

		model.add(Flatten())

		model.add(Dense(128, kernel_regularizer=l2(self.l2_regularizer),
			kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer))
		model.add(Activation('tanh'))

		model.add(Dense(self.labels, kernel_regularizer=l2(self.l2_regularizer),
			kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer))
		model.add(Activation('sigmoid'))

		return(model)

	def get_model(self):
		set_kernel_initializer = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal', seed=None)
		set_bias_initializer = initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)		

		set_kernel_regularizer = regularizers.l1_l2(1e-6)
		set_bias_regularizer = regularizers.l1_l2(1e-6)
		set_activity_regularizer = regularizers.l1_l2(1e-20)

		my_input = Input((self.img_rows, self.img_cols, self.channel_dim))

		conv1 = Conv2D(64, (self.kernel_size, self.kernel_size), padding='same', activation='tanh', kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer, 
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(my_input)
		#pool1 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv1)
		pool1 = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))(conv1)		

		conv2 = Conv2D(128, (self.kernel_size, self.kernel_size), padding='same', activation='tanh', kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer, 
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(pool1)
		#pool2 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv2)
		pool2 = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))(conv2)		

		conv3 = Conv2D(256, (self.kernel_size, self.kernel_size), padding='same', activation='tanh', kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer, 
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(pool2)
		#pool3 = MaxPooling2D(pool_size=(self.pool_size, self.pool_size))(conv3)
		pool3 = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))(conv3)		

		conv4 = Conv2D(512, (self.kernel_size, self.kernel_size), padding='same', activation='tanh', kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer, 
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(pool3)
		pool4 = AveragePooling2D(pool_size=(self.pool_size, self.pool_size))(conv4)

		flat1 = Flatten()(pool4)

		dense1 = Dense(512, activation='tanh', kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer, 
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(flat1)

		my_output = Dense(self.labels, activation='sigmoid', kernel_initializer=set_kernel_initializer, bias_initializer=set_bias_initializer, 
			kernel_regularizer=set_kernel_regularizer, bias_regularizer=set_bias_regularizer, activity_regularizer=set_activity_regularizer)(dense1)

		self.model = Model(inputs=my_input, outputs=my_output)
		return(self.model)

		# Decomissioned
	def test_model(self):
		model = Sequential()

		model.add(Conv2D(32, (self.kernel_size, self.kernel_size), padding='same', input_shape=(self.img_rows, self.img_cols, self.channel_dim)))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))

		model.add(Conv2D(64, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))		
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))

		model.add(Conv2D(128, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(128, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))		

		model.add(Conv2D(256, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(256, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))		
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))

		model.add(Conv2D(512, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(512, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))		

		model.add(Conv2D(1024, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(1024, (self.kernel_size, self.kernel_size), padding='same'))
		model.add(Activation('relu'))
		#model.add(MaxPooling2D(pool_size=(self.pool_size, self.pool_size)))

		model.add(Flatten())

		# one fully connected dense layer of 1024 output dim followed by another dense layer of output dim=1
		model.add(Dense(1024, kernel_regularizer=l2(self.l2_regularizer)))
		model.add(Activation('relu'))
		model.add(Dropout(self.dropout))

		model.add(Dense(self.labels, kernel_regularizer=l2(self.l2_regularizer)))
		model.add(Activation('tanh'))
		#model.add(Activation('relu'))
		#model.add(Dropout(self.dropout))

		return(model)