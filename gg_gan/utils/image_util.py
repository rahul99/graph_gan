from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

def read_animated_images(data_path, size):

	image_data = []

	seed_count = len(os.listdir(data_path))/10
	class_count = 10

	for i in range(seed_count):

		for j in range(class_count):

			image_i_path = data_path + '/Scene' + str(i) + '_' + str(j) + '.png'

			image_i = Image.open(image_i_path).convert("RGB")
			image_i = image_i.resize(size, Image.ANTIALIAS)

			data_i = np.asarray(image_i)

			image_data.append(data_i)

	return(image_data)



def read_video_images(data_path):
	return(None)


def read_other_images(data_path):
	return(None)


def read_image_data(data_path, size, data_type):
	if(data_type in 'animated_data'):
		return(read_animated_images(data_path=data_path, size=size))

	elif(data_type in 'video_data'):
		return(read_video_images(data_path=data_path, size=size))

