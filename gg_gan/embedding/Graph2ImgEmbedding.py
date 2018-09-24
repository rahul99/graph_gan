import sys
sys.path.append('./')
sys.path.append('../')

from gg_gan.utils import graph_util
import numpy as np
from numpy import dot
from scipy.sparse import coo_matrix

def get_Digraph(file_name, directed):
    G = graph_util.loadGraphFromEdgeListTxt(file_name=file_name, directed=directed)
    G = G.to_directed()
    res_pre = 'results/testKarate'
    print 'Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges())
    return(G)

# project the adjacency matrix onto the plane formed by a pair of eigen vectors defined by eigen matrix
def get_projection_planes(adj_matrix, eigen_matrix):
	# assumes each column of the eigen_matrix as one eigen vector
	# project each row of the adjacency matrix onto the eigen-plane
	# each row of the result is a (x,y) cordinate of a point on that plane

	P = [] # append all the planes in this list
	num_of_planes = int(eigen_matrix.shape[1]/2) # if number of eigen vectors are odd, leave the last eigen vector
	for i in range(num_of_planes):
		# take a pair of eigen vectors in the consecutive order
		plane_i = adj_matrix.dot(eigen_matrix[:,(2*i,2*i+1)])
		P.append(plane_i)
	return(P)

def get_start_pos(projection_planes):
	# for every plane, form a grid on the distribution of points. 
	# for that, define a square with starting postion as the minimum (x,y) cordinates
	x_and_y_pos = [np.amin(plane_i, axis=0) for plane_i in projection_planes]
	return(x_and_y_pos)

def get_end_pos(projection_planes):
	# for every plane, form a grid on the distribution of points. 
	# for that, define a square with ending postion as the maximum (x,y) cordinates	
	x_and_y_pos = [np.amax(plane_i, axis=0) for plane_i in projection_planes]
	return(x_and_y_pos)

def get_grid_sizes(projection_planes, condition_image_height):
	# take column wise max for each plane: 1st column corresponds to x-axis, 2nd colm corresponds to y-axis	
	grid_sizes = [np.amax(plane_i, axis=0) for plane_i in projection_planes]
	return(grid_sizes)

def get_graph_as_images(adj_matrix, eigen_matrix, condition_image_height):
	M = [] # final matrix. Stack of 2D frame
	P = get_projection_planes(adj_matrix, eigen_matrix)
	#print("consistency check: dimension of projection matrix is %s" % str(np.shape(P))) # each plane of P is [n_eigen_vector, 2]	

	C_start = [] # position matrix. Stores the starting position of the grid
	C_start = get_start_pos(projection_planes=P)
	#print("consistency check: dimension of starting position is %s" % str(np.shape(C_start))) # dimension is [n_eigen_vector/2,2]	

	C_end = [] # position matrix. Stores the ending position of the grid
	C_end = get_end_pos(projection_planes=P)
	#print("consistency check: dimension of ending position is %s" % str(np.shape(C_end))) # dimension is [n_eigen_vector/2,2]

	total_points_per_plane, _ = np.shape(P[0])

	for k in range(len(P)):
		row_idx = []
		col_idx = []
		value = []
		x_min = C_start[k][0]; y_min = C_start[k][1]
		x_max = C_end[k][0]; y_max = C_end[k][1]		
		delta_x = (x_max - x_min)/float(condition_image_height) # x_increment or grid movement along x
		delta_y = (y_max - y_min)/float(condition_image_height) # y_increment or grid movement along y
		for i in range(condition_image_height):
			for j in range(condition_image_height):
				for ith_point in range(total_points_per_plane):
					x_cord_on_plane=P[k][ith_point,0]
					y_cord_on_plane=P[k][ith_point,1]

					if(x_min + i*delta_x <= x_cord_on_plane and x_cord_on_plane <= x_min + (i+1)*delta_x): # check if the x-cordinate of the plane is in the grid
						if(y_min + j*delta_y <= y_cord_on_plane and y_cord_on_plane <= y_min + (j+1)*delta_y): # check if the y-cordinate of the plane is in the grid
							row_idx.append(j)
							col_idx.append(i)
							value.append(1)

		# Output 2D frames: append each of the 2D images in this set | stored in efficient sparse representation
		frame_k = coo_matrix((value, (row_idx, col_idx)), shape=(condition_image_height, condition_image_height)) # frame_k.toarray() will give the dense matrix
		M.append(frame_k)


	return(M)

