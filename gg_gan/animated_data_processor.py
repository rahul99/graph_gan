import numpy as np
import networkx as nx
import os
from itertools import imap, chain
from operator import sub
from collections import defaultdict
import itertools as iter
import csv
from shutil import copyfile


def pre_process_graph(in_data_path, out_data_path):

	graph_data = read_graph_data(data_path=in_data_path)

	# Check for the missing data for any image. Found following missing data.
	# In "TuplesIdx1_10020.txt": [1696, 1850, 4781, 7908, 8235, 9389, 9791]
	# In "TuplesIdx2_10020.txt": [1355, 3897, 6879, 7099, 9254]
	# Copied from "TuplesIdx1_10020.txt" to "TuplesIdx2_10020.txt" or vice versa as the case may be

	# print list(chain.from_iterable((temp[i] + d for d in xrange(1, diff)) for i, diff in enumerate(imap(sub, temp[1:], temp)) if diff > 1))

	image_count = len(np.unique([dict_i['id'] for dict_i in graph_data]))

	for i in range(image_count):
		graph_i = [dict_i['relation'] for dict_i in graph_data if dict_i['id'] == i]

		outpath = out_data_path + str(i) + '.txt'

		compute_weighted_graph(inp_graph=graph_i, outpath=outpath)


def read_graph_data(data_path):
	data_dict = [] # stores {'image_id':id, 'relation':relation}
	with open(data_path, 'r') as f:
		for line in f:
			image_id = int(line.strip().split()[0])
			relation = line.strip().split()[2:]
			obj_relation = [int(x) for x in relation]

			dict_i = {'id':image_id, 'relation':obj_relation}
			data_dict.append(dict_i)

	return(data_dict)


def write_graph_data(data_list, outpath):

	with open(outpath, 'wb') as f:
		writer = csv.writer(f, delimiter=',')
		for row in data_list:
			writer.writerow(row)



def compute_weighted_graph(inp_graph, outpath):
	# Store dictionary of two keys and a value {'edge_i':e1, 'edge_j':e2, 'weight':ws}
	# key refers to the source and destination edge and value its weight
	weighted_edges_dict = []
	keyfunc = lambda x: x['key']

	for descriptor in inp_graph:

		edge1 = descriptor[0]
		edge2 = descriptor[2]
		weight = 1

		another_edge = {'key':[edge1, edge2], 'value':weight}
		weighted_edges_dict.append(another_edge)

	groups = iter.groupby(sorted(weighted_edges_dict, key=keyfunc), keyfunc)
	#weighted_edges_dict = [{'key':k, 'value':sum(x['value'] for x in g)} for k, g in groups]

	my_list = [[k, [sum(x['value'] for x in g)] ] for k, g in groups]

	my_flattened_list = [sum(element, []) for element in my_list]

	write_graph_data(data_list=my_flattened_list, outpath=outpath)


def pre_process_image(in_data_path, out_data_path, replica_factor=10):

	for i in range(len(os.listdir(in_data_path))):

		src_image = os.path.join(in_data_path, 'Scene' + str(i) + '_0.png')

		for j in range(replica_factor):
			destn_image = os.path.join(out_data_path, 'Scene' + str(i) + '_' + str(j) + '.png')
			copyfile(src_image, destn_image)



def main():

	current_directory=os.getcwd()

	#################################################################################################
	# Data description. Following directory structure must be followed					 			#
	# Create a sub-folder inside the 'data' folder. 									 			#
	# Create following folders insde the sub-folder													#
	# condition_image, target_image, target_graph1, target_graph2 									#
	#################################################################################################	
	condition_image_path = current_directory + '/data/animated_data/condition_image/'
	target_image_path = current_directory + '/data/animated_data/target_image/'

	intrm_graph1_path = current_directory + '/data/intermediate_data/TuplesIdx1_10020.txt'
	intrm_graph2_path = current_directory + '/data/intermediate_data/TuplesIdx2_10020.txt'
	intrm_cond_img_path = current_directory + '/data/intermediate_data/RenderedSeedScenes/'

	target_graph1_path = current_directory + '/data/animated_data/target_graph1/'
	target_graph2_path = current_directory + '/data/animated_data/target_graph2/'


	#################################################################################################
	# Copy each condition image 10 times to a new folder as per the syntax of target_image 			#
	# condition_image and target_image always have one-to-one mapping					 			#
	# target_image and condition_image should always have the same count 							#
	#################################################################################################
	pre_process_image(in_data_path=intrm_cond_img_path, out_data_path=condition_image_path, replica_factor=10)


	#################################################################################################
	# Input Tuple form: [image_idx, sentence_idx, subject_idx, relation_idx, object_idx] 			#
	# Output requirement: [subject_node, object_node, weight]							 			#
	# Here weight is the number of relationships between subject and the object						#
	# Once the graph is converted into adjacency matrix, process the followoing: 					#
	# LLE Embedding, Eigen projection, descretization												#
	#################################################################################################
	pre_process_graph(in_data_path=intrm_graph1_path, out_data_path=target_graph1_path)
	pre_process_graph(in_data_path=intrm_graph2_path, out_data_path=target_graph2_path)

	return(None)