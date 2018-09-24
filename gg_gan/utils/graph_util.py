import sys
sys.path.append('./')
sys.path.append('../')

import networkx as nx
import os


def loadGraphFromEdgeListTxt(file_name, directed):
    with open(file_name, 'r') as f:
        #n_nodes = f.readline()
        #f.readline() # Discard the number of edges
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for line in f:
            edge = line.strip().split(',')
            if len(edge) == 3:
                w = float(edge[2])
            else:
                w = 1.0
            G.add_edge(int(edge[0]), int(edge[1]), weight=w)

    return G


def get_adj_matrix(inp_graph):
    # returns adjacency matrix from edge information
    return(nx.to_scipy_sparse_matrix(inp_graph))
