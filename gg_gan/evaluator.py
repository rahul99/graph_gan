import sys
sys.path.append('./')
sys.path.append('../')

import os
import numpy as np
from keras.models import load_model
from gg_gan.utils import plot_util
from gg_gan.utils import serialize

current_directory = os.getcwd()
condition_data_path = current_directory + '/data/animated_data/serialized/condition_data'
target_data_path = current_directory + '/data/animated_data/serialized/target_data'
target_graph_path = current_directory + '/data/animated_data/serialized/target_graph'

generator_model_path = current_directory + '/data/animated_data/serialized/generator.h5'

def evaluate():
	n_ex=5
	generator_model = load_model(generator_model_path)

	condition_data = serialize.load(condition_data_path)
	target_data = serialize.load(target_data_path)
	target_graph = serialize.load(target_graph_path)


	gen_inps_pair = np.concatenate((condition_data, target_graph), axis=3)
	random_samples = np.random.randint(0,gen_inps_pair.shape[0],size=n_ex)
	condition_image = condition_data[random_samples,:,:,:]
	ground_truth = target_data[random_samples,:,:,:]
	generator_input = gen_inps_pair[random_samples,:,:,:]

	generator_output = generator_model.predict(generator_input)

	plot_util.plot_generator_output(condition_image=condition_image, ground_truth=ground_truth, generator_output=generator_output, n_ex=n_ex)


if __name__ == '__main__':
	evaluate()	
