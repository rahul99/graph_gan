import sys
sys.path.append('./')
sys.path.append('../')

import os,random
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import keras.models as models
from keras.models import Model
from keras.layers import Input,merge

from keras.activations import *
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential

#from keras_adversarial import AdversarialModel
#from keras_adversarial import UnrolledAdversarialOptimizer

from keras.datasets import mnist
import cPickle, random, sys, keras
from keras.models import Model
from keras import backend as K # convert numpy array to tensor

from gg_gan.models import UNet as Generator
from gg_gan.models import CNN as Discriminator
from gg_gan.utils import global_variable

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Define which model (G vs D) needs to be trained at a time
def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

def build_stacked_CGAN(img_rows, img_cols, gen_channel_dim, cond_channel_dim):
	# Build stacked GAN model
	# This is an interesting case of multiple inputs. We use merge layer of keras to pass generator output and conditon image to discriminator
	G_in = Input((img_rows, img_cols, gen_channel_dim))
	G_out = global_variable.myGenerator(G_in)
	#print(G_out)	#Tensor dimension check at graph construction phase

	C_in = Input((img_rows, img_cols, cond_channel_dim)) # conditional image as second input. Merge conditional image and generator output to form discriminator input
	#print(C_in)		#Tensor dimension check at graph construction phase

	D_in = Concatenate(axis=3)([G_out, C_in])
	#print(D_in)	#Tensor dimension check at graph construction phase

	D_out = global_variable.myDiscriminator(D_in)
	#print(D_out)	#Tensor dimension check at graph construction phase

	global_variable.CGAN = Model(inputs=[G_in, C_in], outputs=D_out)


# Pre-train the discriminator network
def pretrain_discriminator(n_ex=10):
	random_samples = np.random.randint(0,global_variable.cond_data.shape[0],size=n_ex)
	batch_of_condition_images = global_variable.cond_data[random_samples,:,:,:]
	generator_input = global_variable.gen_inps_pair[random_samples,:,:,:]

	labels = np.zeros([n_ex,2])
	labels[:,1] = 1 # Optimizing the generator by giving real data labels

	global_variable.CGAN.fit(x=[generator_input, batch_of_condition_images], y=labels)


# alternate (explicit way) without using CGAN
def pretrain_discriminator2(n_ex=10):
	random_samples = np.random.randint(0,global_variable.cond_data.shape[0],size=n_ex)
	batch_of_condition_images = global_variable.cond_data[random_samples,:,:,:]
	generator_input = global_variable.gen_inps_pair[random_samples,:,:,:]
	batch_of_real_pairs = global_variable.dis_inps_pair[random_samples,:,:,:]

	generator_output = global_variable.myGenerator.predict(generator_input)

	# The fake input to the discriminator is concatenated with the condition image along the channel dimension
	# This is to ensure that the GAN does not learn just the indentity mapping
	batch_of_fake_pairs = np.concatenate((generator_output, batch_of_condition_images), axis=3)

	discriminator_input = np.concatenate((batch_of_real_pairs, batch_of_fake_pairs), axis=0)

	data_count = discriminator_input.shape[0]
	labels = np.zeros([data_count,2])
	labels[:data_count,1] = 1
	labels[data_count:,0] = 1
	global_variable.myDiscriminator.trainable=True
	#make_trainable(global_variable.myDiscriminator,True)
	global_variable.myDiscriminator.fit(x=discriminator_input, y=labels, epochs=1, batch_size=10)
	y_hat = global_variable.myDiscriminator.predict(discriminator_input)



def train(inp_cond_data, inp_trgt_data, inp_trgt_graph):

	#################################################################################
	#																				#
	#            		Declare global variables and assign values					#
	#																				#
	#################################################################################
	# set up loss storage vector
	global_variable.losses = {"d":[], "g":[]}

	# Conditional data is additional input to the model. It is merged with real input to the discriminator
	# It is also merged with generated output to produce fake input pair to the discriminator.
	global_variable.cond_data = inp_cond_data
	global_variable.trgt_data = inp_trgt_data

	# Build input to the generator [condition data + target graph]
	global_variable.gen_inps_pair = np.concatenate((global_variable.cond_data, inp_trgt_graph), axis=3)

	# Build input to the descriminator [[condition data + target data as real input], [condition data + generator output as fake input]]
	global_variable.dis_inps_pair = np.concatenate((global_variable.cond_data, global_variable.trgt_data), axis=3)


	#################################################################################
	#																				#
	#   		       Build stacked GAN framework and compile models				#
	#																				#
	#################################################################################
	img_rows = global_variable.cond_data.shape[1]
	img_cols = global_variable.cond_data.shape[2]
	cond_channel_dim = global_variable.cond_data.shape[3]
	gen_channel_dim = cond_channel_dim + inp_trgt_graph.shape[3]

	# Freeze weights in the discriminator for stacked training
	#make_trainable(global_variable.myDiscriminator, False)
	global_variable.myDiscriminator.trainable=False
	build_stacked_CGAN(img_rows=img_rows, img_cols=img_cols, gen_channel_dim=gen_channel_dim, cond_channel_dim=cond_channel_dim)

	# Compile generator, discriminator and CGAN models
	global_variable.myGenerator.compile(loss='binary_crossentropy', optimizer=global_variable.gen_opt)
	global_variable.myGenerator.summary()
	global_variable.myDiscriminator.compile(loss='binary_crossentropy', optimizer=global_variable.dis_opt)
	global_variable.myDiscriminator.summary()
	global_variable.CGAN.compile(loss='binary_crossentropy', optimizer=global_variable.gen_opt)
	global_variable.CGAN.summary()


	'''
	#################################################################################
	#																				#
	#           AdHoc. Trying adversarial network example 							#
	#																				#
	#################################################################################

	final_model = AdversarialModel(base_model=global_variable.CGAN, player_params=[global_variable.myGenerator.trainable_weights,
		global_variable.myDiscriminator.trainable_weights], player_names=["global_variable.myGenerator", "global_variable.myDiscriminator"], loss='binary_crossentropy')

	final_model.adversarial_compile(adversarial_optimizer=UnrolledAdversarialOptimizer(), player_optimizers=[global_variable.gen_opt, global_variable.dis_opt])
	'''


	#################################################################################
	#																				#
	#           Train the network adaptively with different learning rates 			#
	#																				#
	#################################################################################
	# Pre-train discriminator to have initial guess of the weights
	pretrain_discriminator2(n_ex=500)

	# Train at standard learning rate with large epoches
	train_for_n(num_epoch=100, plt_frq=500, batch_size=64)

	# Trained at reduced learning rate
	K.set_value(global_variable.gen_opt.lr, 1e-4)
	K.set_value(global_variable.dis_opt.lr, 1e-3)
	train_for_n(num_epoch=100, plt_frq=500, batch_size=32)	

	# Trained at reduced learning rate again
	K.set_value(global_variable.gen_opt.lr, 1e-5)
	K.set_value(global_variable.dis_opt.lr, 1e-4)
	train_for_n(num_epoch=100, plt_frq=500, batch_size=16)	


def train_for_n(num_epoch, plt_frq, batch_size):

	global lossess
	for iter in tqdm(range(num_epoch)):

		random_samples = np.random.randint(0, global_variable.gen_inps_pair.shape[0], size=batch_size)

		batch_of_condition_images = global_variable.cond_data[random_samples,:,:,:]
		generator_input = global_variable.gen_inps_pair[random_samples,:,:,:]
		batch_of_real_pairs = global_variable.dis_inps_pair[random_samples,:,:,:]

		# Extract generated images
		generator_output = global_variable.myGenerator.predict(generator_input)
		batch_of_fake_pairs = np.concatenate((generator_output, batch_of_condition_images), axis=3)

		# Train discriminator on original pair versus fake pair
		discriminator_input = np.concatenate((batch_of_real_pairs, batch_of_fake_pairs), axis=0)
		labels = np.zeros([2*batch_size,2])
		labels[0:batch_size,1] = 1 # real data labels [0,1]
		labels[batch_size:,0] = 1 # fake data labels [1,0]

		global_variable.myDiscriminator.trainable=True
		#make_trainable(myDiscriminator,True)
		d_loss  = global_variable.myDiscriminator.train_on_batch(discriminator_input,labels)
		global_variable.losses["d"].append(d_loss)

		# Train GAN stack on generator input to non-generated output class
		random_samples = np.random.randint(0, global_variable.gen_inps_pair.shape[0], size=batch_size)
		#generator_input = global_variable.gen_inps_pair[random_samples,:,:,:]
		#batch_of_condition_images = global_variable.cond_data[random_samples,:,:,:]
		labels = np.zeros([batch_size,2])
		labels[:,1] = 1 # Optimizing the generator by giving real data labels
		global_variable.myDiscriminator.trainable=False
		g_loss = global_variable.CGAN.train_on_batch([generator_input, batch_of_condition_images], labels)
		print("g_loss is: ", g_loss, "and d_loss is: ", d_loss)
		global_variable.losses["g"].append(g_loss)
