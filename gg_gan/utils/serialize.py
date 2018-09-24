import pickle

def dump(data_path, data):
	with open(data_path, 'wb') as f:
		pickle.dump(data, f)


def load(data_path):
	with open(data_path, 'rb') as f:
		data = pickle.load(f)

	return(data)
