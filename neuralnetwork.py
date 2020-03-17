simport random
import numpy as np
import time

def sigmoid(Z):
    '''Sigmoid function'''
    return 1 / (1 + np.exp(-Z))

def sigmoid_prime(Z):
    '''Derivative of the sigmoid function'''
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu(Z):
    '''ReLU function'''
    return Z * (Z > 0)

def relu_prime(Z):
    '''Derivative of the ReLU function'''
    return 1 * (Z > 0)

def select_activation(activation):
    # Set the activation function
    if activation == "relu":
        return relu
    elif activation == "sigmoid":
        return sigmoid
    else:
        raise Exception("Unsupported activation function")

def select_prime(prime):
    # Set the activation function
    if prime == "relu":
        return relu_prime
    elif prime == "sigmoid":
        return sigmoid_prime
    else:
        raise Exception("Unsupported activation function")

def cost_func_derivative(a, y):
    return (a - y)

class NeuralNetwork:
    def __init__(self, layers, seed = 2):
        '''Initialise the network with random weights and biases.
        The activation functions for each layer are stored as an
        attribute.'''
        np.random.seed(seed)
        self.num_layers = len(layers)
        # Initiliase random values for the weights and biases in each layer
        self.weights = [np.random.randn(y[0], x[0]) 
                        for x, y in zip(layers[:-1], layers[1:])]  
        self.biases = [np.random.randn(y[0], 1) for y in layers[1:]]
        # Set activations for each layer
        self.activations = [x[1] for x in layers[1:]]

    def forward_prop(self, a):
        '''Take an input column vector and propagates through
        the network to give the output.'''
        # Iterate through layers in the network
        for w, b, activation in zip(self.weights, self.biases, self.activations):
            func = select_activation(activation)
            # Compute z for the current layer
            z = np.dot(w, a) + b
            a = func(z)
        return a

    def backward_prop(self, x, y):
        '''This function performs the back propagation algorithm 
        for each training example. Firstly this function performs 
        the forward propagation and stores the activations and z 
        values for each layer. The function then backpropagates the 
        error in the prediction, finding the error for each layer, 
        using this to calculate the partial derivative of the cost 
        function w.r.t. the weights and biases in each layer.'''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # Set first activation as input
        a = x
        As = [x]
        Zs = []
        # Forward propagation
        for w, b, activation in zip(self.weights, self.biases, self.activations):
            func = select_activation(activation)
            # Compute and store z for the current layer
            z = np.dot(w, a) + b
            Zs.append(z)
            # Compute and store a for the current layer
            a = func(z)
            As.append(a)
        # Back propagation
        prime = select_prime(self.activations[-1])
        delta = cost_func_derivative(As[-1], y) * prime(Zs[-1])
        # Set the derivative of cost function w.r.t. the weights and biases for
        # the output layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, As[-2].T)
        for idx in range(2, self.num_layers):
            prime = select_prime(self.activations[-idx])
            # Calculate the error in each layer
            delta = np.dot(self.weights[-idx + 1].T, delta) * prime(Zs[-idx])
            # Store the derivative of cost function w.r.t. the weights and biases
            nabla_b[-idx] = delta
            nabla_w[-idx] = np.dot(delta, As[-idx -1].T)
        return nabla_w, nabla_b

    def update_weights(self, mini_batch, eta):
        '''Store the partial derivatives of the cost function 
        w.r.t. the weights and biases from each training example 
        for each layer, average over the errors and use this to 
        calculate the overall adjustment required for each of the 
        weights and biases.''' 
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        m = len(mini_batch)
        for x, y in mini_batch:
            # Calculate the error at each layer for each training example
            delta_nabla_w, delta_nabla_b = self.backward_prop(x, y)
            # Store the summation of these errors for each layer
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # For each layer adjust the weights and biases accordingly
        self.weights = [w - (eta / m) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb 
                        for b, nb in zip(self.biases, nabla_b)]

    def train_network(self, training_data, epochs, batch_size, eta, test_data=None):
        '''This function divides the training data into
        batches and then passes each batch to the weight
        updating function. There is an optional functionality
        where, if test data is provided, the accuracy of the
        classifications is checked each epoch.'''
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        start = time.time()
        for j in range(epochs):
            # Shuffle training data
            random.shuffle(training_data)
            # Divide the training data into batches
            mini_batches = [training_data[k:k + batch_size] 
                            for k in range(0, n, batch_size)]
            # Update the weights for each batch
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, eta)
            # Check the classification accuracy
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j+1, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j+1))
        end = time.time()
        print("Time to train: ", end - start)

    def evaluate(self, test_data):
        '''Checks the current classification accuracy by
        feeding the test images forward and then comparing
        the results with the test labels.'''
        # Take the position of the largest value as the classification
        test_results = [(np.argmax(self.forward_prop(x)), y)
                        for x, y in test_data]
        # Convert the true/false values to integers and sum over the results
        # to give the number of correctly classified images
        return sum(int(x==y) for x, y in test_results)

if __name__ == "__main__":
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.import_data()
    layers = [[784, "input"], [32, "sigmoid"], [16, "relu"], [10, "sigmoid"]]
    net = NeuralNetwork(layers)
    net.train_network(training_data, epochs=20, batch_size=100, eta=2, test_data=test_data)