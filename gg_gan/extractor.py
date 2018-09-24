import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
import pickle
import numpy as np

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

def get_cifar_data():
	# The data, shuffled and split between train and test sets:
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	return x_train

	print('x_train shape:', x_train.shape) # ('x_train shape:', (50000, 32, 32, 3))
	channel_dim = x_train.shape[3]
	print("channel dim is %s" %channel_dim)

	print("condition image + trgt graph")
	conditon_image = x_train
	trgt_graph = x_train

	gen_inp = np.concatenate((conditon_image, trgt_graph), axis=3)

	print("shape of gen_inp is:", gen_inp.shape)

	exit(0)


	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	input_shape=x_train.shape[1:]
	print(input_shape)

	# Convert class vectors to binary class matrices.
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)