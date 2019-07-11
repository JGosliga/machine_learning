import numpy as np

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