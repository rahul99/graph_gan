import sys
sys.path.append('./')
sys.path.append('../')

from gg_gan.utils import graph_util
from gg_gan.embedding import Graph2ImgEmbedding, LocallyLinearEmbedding
import os
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr


def process_graph_data(data_path, image_size, eigen_space):

	# constructor defines the dimension of eigen space to be retained
	embedding = LocallyLinearEmbedding(eigen_space)

	graph_data = []

	for file_idx in range(len(os.listdir(data_path))):

		file_path = os.path.join(data_path, str(file_idx) + '.txt')
		#file_path = os.path.join(data_path, '127.txt')

		diGraph = graph_util.loadGraphFromEdgeListTxt(file_name=file_path, directed=True)

		# Embedding essentially means that use LL and then store the top-k eigen vectors
		embedding.learn_embedding(graph=diGraph, edge_f=None, is_weighted=True, no_python=True)

		adj_matrix = graph_util.get_adj_matrix(inp_graph=diGraph)
		adj_matrix.astype(float)

		eigen_matrix = embedding.get_embedding()
		eigen_matrix.astype(float)
		if(isspmatrix_csr(eigen_matrix)):
			eigen_matrix = eigen_matrix.toarray()

		# coco-sparse matrix representation for efficient storage
		graph_i_frame_list = Graph2ImgEmbedding.get_graph_as_images(adj_matrix=adj_matrix, eigen_matrix=eigen_matrix, condition_image_height=image_size)

		graph_i_frames = np.dstack([x.toarray() for x in graph_i_frame_list])

		graph_data.append(graph_i_frames)

	return(graph_data)
