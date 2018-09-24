# entry point to the project. All methods are called from this script

import sys
sys.path.append('./')
sys.path.append('../')

from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from gg_gan.embedding import graph_driver

from gg_gan.utils import graph_util, image_util
from gg_gan.utils import global_variable
from gg_gan.utils import plot_util
from gg_gan.utils import serialize

from gg_gan.models import UNet as Generator
from gg_gan.models import CNN as Discriminator

from gg_gan.training import trainer

from gg_gan import animated_data_processor as data_processor
from gg_gan import evaluator



#################################################################################################
# Component 1: Pre-process data as per the type of the data.									#
# Write a data_processor specific to the data 													#
# Store the data inside 'data' folder by creating a sub folder for the data.					#
#################################################################################################
data_descriptor = 'animated_data'

current_directory = os.getcwd()
source_data_path = os.path.join(current_directory, 'data/' + data_descriptor)
condition_data_path = os.path.join(source_data_path, 'condition_image')
target_data_path = os.path.join(source_data_path, 'target_image')


# if the source data in animated_data, merge the two type of graphs generated from two different sentences
if(data_descriptor=='animated_data'):

	# Serialize the data. So the following computationaly expensive step happens only one time
	serialized_path = source_data_path + '/serialized'
	if not os.path.exists(serialized_path):

		#####################################################################
		# Pre process TuplesIdx1_10020.txt into a weighted graph (i,j,w) 	#
		# Pre process TuplesIdx2_10020.txt into a weighted graph (i,j,w)	#
		# Pre process RenderedSeesScenes: 									#
		#	Copy 10 times to match the target image (one-one mapping) 		#
		#####################################################################
		data_processor.main()


		#####################################################################
		# Process graph_data1 to generated graph as image frames		 	#
		# Process graph_data2 to generated graph as image frames			#
		# Merge the two graph frames along channel dimension 				#
		#####################################################################
		target_graph1_path = os.path.join(source_data_path, 'target_graph1')
		target_graph2_path = os.path.join(source_data_path, 'target_graph2')

		# dimnesion = [10020, 32, 32, 1] = [no of samples, row, col, channel]
		target_graph1 = graph_driver.process_graph_data(data_path=target_graph1_path, image_size=64, eigen_space=2)
		target_graph1 = np.asarray(target_graph1)

		target_graph2 = graph_driver.process_graph_data(data_path=target_graph2_path, image_size=64, eigen_space=2)
		target_graph2 = np.asarray(target_graph2)

		target_graph = np.concatenate([target_graph1, target_graph2], axis=3)


		#####################################################################
		# Create numpy array [n_samples, row, col, 3] for condition images	#
		# Create numpy array [n_samples, row, col, 3] for target images		#
		#####################################################################
		condition_data = image_util.read_image_data(data_path=condition_data_path, size=[64,64], data_type=data_descriptor)
		target_data = image_util.read_image_data(data_path=target_data_path, size=[64,64], data_type=data_descriptor)

		condition_data = np.asarray(condition_data)
		target_data = np.asarray(target_data)


		#####################################################################
		# Serialize the data for fast loading in subsequent runs		 	#
		#####################################################################
		os.makedirs(serialized_path)

		data_path = serialized_path + '/condition_data'
		serialize.dump(data_path=data_path, data=condition_data)
		data_path = serialized_path + '/target_data'
		serialize.dump(data_path=data_path, data=target_data)
		data_path = serialized_path + '/target_graph'
		serialize.dump(data_path=data_path, data=target_graph)


#################################################################################################
# Component 2: Read serialized input data 														#
# Condition image as [n_samples, 32, 32, 3] 													#
# target image as [n_samples, 32, 32, 3] 														#
# target graph as [n_samples, 32, 32, 2] 														#
#################################################################################################
data_path = serialized_path + '/condition_data'
condition_data = serialize.load(data_path)
data_path = serialized_path + '/target_data'
target_data = serialize.load(data_path)
data_path = serialized_path + '/target_graph'
target_graph = serialize.load(data_path)

print condition_data.shape
print target_data.shape
print target_graph.shape


#################################################################################################
# Component 3: #Training GAN Network															#
# num_samples = condition_data.shape[0] = target_data.shape[0] = target_graph.shape[0]			#
# condition_data = [num_samples, row,col,3]														#
# target_data = [num_samples, row,col,3] 														#
# target_graph = [num_samples, row,col,2]	 													#
#################################################################################################

global_variable.init()

img_rows = condition_data.shape[1]
img_cols = condition_data.shape[2]
gen_channel_dim_in = condition_data.shape[3] + target_graph.shape[3]
gen_channel_dim_out = condition_data.shape[3]
dis_channel_dim_in = condition_data.shape[3] + target_data.shape[3]

generator = Generator(my_img_rows=img_rows, my_img_cols=img_cols, my_channel_dim=gen_channel_dim_in, my_out_channel_dim=gen_channel_dim_out, my_kernel_size=3, my_pool_size=2)
global_variable.myGenerator = generator.get_model()
discriminator = Discriminator(my_img_rows=img_rows, my_img_cols=img_cols, my_channel_dim=dis_channel_dim_in, my_kernel_size=3, my_pool_size=2, my_l2_regularizer=0.001, my_dropout=0.25, my_labels=2) #takes pair of RGB images. Hence channel_dim=6
global_variable.myDiscriminator = discriminator.get_model()

print("Training Model")

trainer.train(inp_cond_data=condition_data, inp_trgt_data=target_data, inp_trgt_graph=target_graph)


#################################################################################################
# Component 4: Save the trained Generator network 				 												#
# (i) Plot loss curves for the generator and the descriminator 									#
# (ii) Plot some generated images and compare with ground truth against the condition			#
#################################################################################################
generator_data_path = serialized_path + '/generator.h5'

global_variable.myGenerator.save(generator_data_path)

plot_util.plot_gan_loss()
plt.show()

print("execution complete!")


#################################################################################################
# Component 5: Evaluate the GAN framework 				 												#
# (i) Plot loss curves for the generator and the descriminator 									#
# (ii) Plot some generated images and compare with ground truth against the condition			#
#################################################################################################
# alternatively, run the evaluator script independently of the driver 
#evaluator.evaluate()
