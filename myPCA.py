import numpy as np

def my_PCA(data, number_of_components):
	# centre incoming data around the origin
	centred_data = centre_data(data)
	return centred_data

def centre_data(data):
	averages = []
	for column in data.T:
		averages.append(sum(column)/column.size)
	return data - np.array(averages)

if __name__ == "main":
	pass