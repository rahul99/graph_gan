disp_avlbl = True
from os import environ
if 'DISPLAY' not in environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
from time import time
import pdb

import sys
sys.path.append('./')
sys.path.append('../')

from gg_gan.utils import graph_util


class LocallyLinearEmbedding:
	# d specifies the dimension of eigen space to be retained for the graph
	def __init__(self, d):
		self._d = d
		self._method_name = 'lle_svd'
		self._X = None

	def get_method_name(self):
		return self._method_name

	def get_method_summary(self):
		return '%s_%d' % (self._method_name, self._d)

	def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
		if not graph and not edge_f:
			raise Exception('graph/edge_f needed')
		if not graph:
			graph = graph_util.loadGraphFromEdgeListTxt(edge_f)
		graph = graph.to_undirected()
		t1 = time()

		A = nx.to_scipy_sparse_matrix(graph)

		# Check if the dimension of A is 2x2 in which case, the eigen vector is not defined
		flag = (A.shape == (2,2))
		if(flag):
			self._X = A
			return self._X

		#A_normed = normalize(A, norm='l2', axis=1, copy=False)
		I = sp.eye(graph.number_of_nodes())
		I_min_A = I - A
		#u, s, vt = lg.svds(I_min_A, k=self._d+1, which='LM')
		u, s, vt = lg.svds(A, k=self._d, which='LM')

		t2 = time()
		self._X = vt.T
		return self._X, (t2-t1)

	def get_embedding(self):
		return self._X

	def get_edge_weight(self, i, j):
		return np.exp(-np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2))

	def get_reconstructed_adj(self, X=None, node_l=None):
		if X is not None:
			node_num = X.shape[0]
			self._X = X
		else:
			node_num = self._node_num
		adj_mtx_r = np.zeros((node_num, node_num)) # G_r is the reconstructed graph
		for v_i in range(node_num):
			for v_j in range(node_num):
				if v_i == v_j:
					continue
				adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
		return adj_mtx_r

if __name__ == '__main__':
	# load Zachary's Karate graph
	edge_f = 'data/karate.edgelist'
	G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
	G = G.to_directed()
	res_pre = 'results/testKarate'
	print 'Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges())
	t1 = time()
	embedding = LocallyLinearEmbedding(2)
	embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
	print 'Graph Factorization:\n\tTraining time: %f' % (time() - t1)

	#viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
plt.show()