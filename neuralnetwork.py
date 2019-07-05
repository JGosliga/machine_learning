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
    
    def forward_prop_single_layer(self, A, idx):
        # Calculate Z for the current layer
        Z = np.dot(self.weights[idx], A) + self.biases[idx]
        # Select activation function for the current layer
        if self.activations[idx] is "relu":
            activation_func = relu
        elif self.activations[idx] is "sigmoid":
            activation_func = sigmoid
        else:
            raise Exception('Non-supported activation function')
        # Return the current Z and the result of the activation function
        return activation_func(Z), Z
        
    def full_forward_prop(self, X):
        # Convert the input values to a numpy array
        # Reshape X into a column vector
        X = X.reshape(X.shape[0],1) / X.max()
        # Set X as input to first layer
        A_curr = X
        # Store the input and output for each layer
        self.memory = {}
        # Perform the forward propagation for each layer
        for idx in range(self.number_of_layers):
            # Perform forward propagation using the input to the current layer
            A_prev = A_curr
            A_curr, Z_curr = self.forward_prop_single_layer(A_prev, idx)
            # Store the input from the current layer, as well as the output
            self.memory["A" + str(idx)] = A_prev
            self.memory["Z" + str(idx + 1)] = Z_curr
        return A_curr

    def cost_function(self, Y_hat, Y):
        # Calculate squared errors in the predictions
        squared_errors = (Y_hat - Y) ** 2
        return np.sum(squared_errors)

    def back_prop_single_layer(self, dC_dA, Z_curr, A_prev, idx):
        # Number of neurons to average over
        n = A_prev.shape[0]
        # Select the correct activation function for the current layer
        if self.activations[idx] is "relu":
            activation_func_dif = relu_dif
        elif self.activations[idx] is "sigmoid":
            activation_func_dif = sigmoid_dif
        else:
            raise Exception('Non-supported activation function')
        # Calculate the derivative of the cost function wrt z
        dC_dZ = dC_dA * activation_func_dif(Z_curr)
        # Calculate the derivative of the cost function wrt the weights 
        dC_dW = np.dot(dC_dZ, A_prev.T) / n
        # Calculate the derivative of the cost function wrt the biases
        dC_db = np.sum(dC_dZ, axis=1, keepdims=True) / n
        # Calulcate the derivative of the cost function wrt to the layer input for next layer
        dC_dA_next = np.dot(self.weights[idx].T, dC_dZ)
        # Return the derivate of the cost function wrt to the input for the next layer, as well as
        # the adjustments to the weights and biases
        return dC_dA_next, dC_dW, dC_db

    def full_backward_prop(self, Y_hat, Y):
        # Convert the true y values to a numpy array
        Y = np.array(Y)
        # This is the initial derivative of the cost function wrt to the output from the last layer
        dC_dA_next = 2 * (Y_hat - Y)
        for idx in reversed(range(self.number_of_layers)):
            # Use the value for dC_dA from the previous layer
            dC_dA = dC_dA_next
            # Retrieve the layer inputs and outputs from the forward propagation step
            A_prev = np.array(self.memory["A" + str(idx)])
            Z_curr = self.memory["Z" + str(idx + 1)]
            # Propagate backwards for each layer
            dC_dA_next, dC_dW, dC_db = self.back_prop_single_layer(dC_dA, Z_curr, A_prev, idx)
            # Adjust weights and biases for the current layer
            self.store_adjustments(dC_dW, dC_db, idx)
    
    def store_adjustments(self, dC_dW, dC_db, idx):
        self.weight_adjust[idx] += dC_dW
        self.biases_adjust[idx] += dC_db

    def parameter_optimisation(self, steps):
        # Controls how quickly the algorithm moves through the gradient descent
        learning_rate = 3
        # Move in the opposite direction from the steepest ascent
        for idx, weights in enumerate(self.weights):
            self.weights[idx] -= learning_rate / steps * np.array(self.weight_adjust[idx])
            self.biases[idx] -= learning_rate /steps * np.array(self.biases_adjust[idx])
            # for idx2, unit in enumerate(weights):
            #     weights_norm = np.sqrt(sum(unit ** 2))
            #     if weights_norm > 3:
            #         self.weights[idx][idx2] = np.array(unit) / unit.max()
            #         self.biases[idx] = np.array(self.biases[idx]) / unit.max()
            
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