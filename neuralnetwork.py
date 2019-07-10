import numpy as np
import struct as st
import sys, os

def sigmoid(Z):
    '''Sigmoid function'''
    return 1 / (1 + np.exp(-Z))

def sigmoid_dif(Z):
    '''Derivative of the sigmoid function'''
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu(Z):
    '''ReLU function'''
    return Z * (Z > 0)

def relu_dif(Z):
    '''Derivative of the ReLU function'''
    return Z > 0

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def import_data():
    # Obtain the current directory for the python file
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    filename = {'training_images' : 'train-images-idx3-ubyte' ,
                'training_labels' : 'train-labels-idx1-ubyte',
                'test_images'     : 't10k-images-idx3-ubyte',
                'test_labels'     : 't10k-labels-idx1-ubyte'}

    # Extract the training images
    train_images_file = open(os.path.join(__location__, filename['training_images']),'rb')
    # Skip first 4 bytes
    train_images_file.seek(4)
    # Find information regarding number and size of images
    nImg = st.unpack('>I',train_images_file.read(4))[0] # Number of images
    nR = st.unpack('>I',train_images_file.read(4))[0] # Number of rows
    nC = st.unpack('>I',train_images_file.read(4))[0] # Numbe of columns
    nBytesTotal = nImg*nR*nC*1 # Each pixel data is 1 byte
    # Organise images into 784 pixel long row vectors
    train_images_array = 1 - np.asarray(st.unpack('>'+'B'*nBytesTotal,
                                            train_images_file.read(nBytesTotal))).reshape((nImg,nR*nC)) / 255

    # Extract the training labels
    train_labels_file = open(os.path.join(__location__, filename['training_labels']),'rb')
    # Skip first 4 bytes again
    train_labels_file.seek(4)
    nLbls = st.unpack('>I',train_labels_file.read(4))[0] # Number of labels
    nBytesTotal = nLbls*1 # Each label is 1 byte
    train_labels_array = np.array(st.unpack('>'+'B'*nBytesTotal,
                                    train_labels_file.read(nBytesTotal))).reshape(nLbls)

    # Extract the test images
    test_images_file = open(os.path.join(__location__, filename['test_images']),'rb')
    test_images_file.seek(4)
    # Find information regarding images
    nImg = st.unpack('>I',test_images_file.read(4))[0] # Number of images
    nR = st.unpack('>I',test_images_file.read(4))[0] # Number of rows
    nC = st.unpack('>I',test_images_file.read(4))[0] # Numbe of columns
    nBytesTotal = nImg*nR*nC*1 # Each pixel data is 1 byte
    # Organise images into 784 pixel long row vectors
    test_images_array = 1 - np.asarray(st.unpack('>'+'B'*nBytesTotal, 
                                        test_images_file.read(nBytesTotal))).reshape((nImg,nR*nC)) / 255

    # Extract the training labels
    test_labels_file = open(os.path.join(__location__, filename['test_labels']),'rb')
    # As before, skip the first 4 bytes
    test_labels_file.seek(4)
    nLbls = st.unpack('>I',test_labels_file.read(4))[0] # Number of labels
    nBytesTotal = nLbls*1 # Each label is 1 byte
    test_labels_array = np.array(st.unpack('>'+'B'*nBytesTotal,test_labels_file.read(nBytesTotal))).reshape(nLbls)

    return zip_data(train_images_array, train_labels_array, test_images_array, test_labels_array)

def zip_data(train_images_array, train_labels_array, test_images_array, test_labels_array):
    training_inputs = [np.reshape(x, (784, 1)) for x in train_images_array]
    training_results = [vectorized_result(y) for y in train_labels_array]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in test_images_array]
    test_data = zip(test_inputs, test_labels_array)

    return training_data, test_data
    
def convert_labels_to_input(labels):
    # Set classification vector according to the labels
    Y = np.zeros((labels.shape[0],10,1))
    for idx, y in enumerate(Y):
        clss = labels[idx]
        # Set the corresponding position in output vector to reflect class label
        y[clss] = 1
    return Y

class NeuralNetwork:
    def __init__(self, architecture, seed = 1):
        np.random.seed(seed)
        self.number_of_layers = len(architecture)
        self.weights = list(np.zeros(self.number_of_layers))
        self.biases = list(np.zeros(self.number_of_layers))
        self.weight_adjust = list(np.zeros(self.number_of_layers))
        self.biases_adjust = list(np.zeros(self.number_of_layers))
        self.activations = []

        # Iterate over each layer specified in the architecture
        for idx, layer in enumerate(architecture):
            # Extract layer dimensions
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            # Initiliase small random values for the weights and biases in each layer
            self.weights[idx] = 0.1 * np.random.randn(layer_output_size, layer_input_size)
            self.biases[idx] = 0.1 * np.random.randn(layer_output_size, 1)
            # Set activations for each layer
            layer_activation = layer["activation"]
            self.activations.append(layer_activation)
            
    def train_network(self, X, labels, epochs):
        # Iterates the forward and backward propagation steps to train on the data given
        steps = X.shape[0] - 50000
        for j in range(epochs):
            for i in range(steps):
                Y_hat = self.full_forward_prop(X[i])
                self.full_backward_prop(Y_hat, Y[i])
            self.parameter_optimisation(steps)
            print("Current epoch:", j + 1)

if __name__ == "__main__":
    architecture = [{"input_dim" : 784, "output_dim" : 30, "activation" : "relu"},
                    {"input_dim" : 30, "output_dim" : 10, "activation" : "sigmoid"}]
    net = NeuralNetwork(architecture)

    training_data, test_data = import_data()
    print(test_data)