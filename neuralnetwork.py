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

class NeuralNetwork:
    def __init__(self, layers, seed = 1):
        np.random.seed(seed)
        self.num_layers = len(layers)
        self.weights = list(np.zeros(self.num_layers))
        self.biases = list(np.zeros(self.num_layers))
        self.activations = []

        # Iterate over each layer specified in the layers
        for idx, layer in enumerate(layers):
            # Initiliase small random values for the weights and biases in each layer
            if idx != (self.num_layers - 1):
                self.weights[idx] = 0.1 * np.random.randn(layers[idx + 1][0], layer[0])
            if idx != 0:
                self.biases[idx] = 0.1 * np.random.randn(layer[0], 1)
            # Set activations for each layer
            self.activations.append(layer[1])

    def train_network(self, X, labels, epochs):
        # Iterates the forward and backward propagation steps to train on the data given
        steps = X.shape[0] - 50000
        # for j in range(epochs):
        #     for i in range(steps):
        #         Y_hat = self.full_forward_prop(X[i])
        #         self.full_backward_prop(Y_hat, Y[i])
        #     self.parameter_optimisation(steps)
        #     print("Current epoch:", j + 1)

if __name__ == "__main__":
    layers = [[2, "input"], [3, "relu"], [1, "sigmoid"]]
    net = NeuralNetwork(layers)
    print(net.weights)
    print(net.biases)
    print(net.activations)