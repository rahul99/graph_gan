import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from gg_gan.utils import global_variable

def plot_gan_loss():
	plt.figure(figsize=(10,8))
	plt.plot(global_variable.losses["d"], label='discriminitive loss')
	plt.plot(global_variable.losses["g"], label='generative loss')
	plt.legend()

def plot_generator_output(condition_image, ground_truth, generator_output, n_ex=5):

	# Plot generated output
	plt.figure()
	GS = gridspec.GridSpec(3, n_ex, top=0.9, bottom=0.1, right=0.95, left=0.05, hspace=0.2, wspace=0.2)

	cond_counter=0; truth_counter=0; gen_counter=0
	# grid moves from left to right in a row and for all rows
	for grid in GS:
		ax = plt.subplot(grid)

		# Plot all condition images in first row
		if(cond_counter<n_ex):
			ax.imshow(condition_image[cond_counter,:,:,:])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_aspect('auto')
			ax.set_title('Condition Image ' + str(cond_counter+1))
			cond_counter = cond_counter + 1

		# Plot all target images in second row
		elif(truth_counter<n_ex):
			ax.imshow(ground_truth[truth_counter,:,:,:])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_aspect('auto')
			ax.set_title('Target Image ' + str(truth_counter+1))			
			truth_counter = truth_counter + 1

		# Plot all generated images in third row
		else:
			ax.imshow(generator_output[gen_counter,:,:,:])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_aspect('auto')
			ax.set_title('Generated Image ' + str(gen_counter+1))
			gen_counter = gen_counter + 1
	plt.show()