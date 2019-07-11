import random
import numpy as np

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
    return Z > 0

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
        np.random.seed(seed)
        self.num_layers = len(layers)
        # Initiliase random values for the weights and biases in each layer
        self.weights = [np.random.randn(y[0], x[0]) 
                        for x, y in zip(layers[:-1], layers[1:])]  
        self.biases = [np.random.randn(y[0], 1) for y in layers[1:]]
        # Set activations for each layer
        self.activations = [x[1] for x in layers[1:]]

    def forward_prop(self, a):
        # Iterate through layers in the network
        for w, b, activation in zip(self.weights, self.biases, self.activations):
            func = select_activation(activation)
            # Compute z for the current layer
            z = np.dot(w, a) + b
            a = func(z)

    def backward_prop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # Set first activation as input
        a = x
        As = [x]
        Zs = []
        # Forward propagation
        for w, b, activation in zip(self.weights, self.biases, self.activations):
            func = select_activation(activation)
            # Compute z for the current layer
            z = np.dot(w, a) + b
            # The acitvations are offset by 1 so As[0] is first layer activation
            Zs.append(z)
            a = func(z)
            As.append(a)
        # Back propagation
        prime = select_prime(self.activations[-1])
        delta = cost_func_derivative(As[-1], y) * prime(Zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, As[-2].T)
        for idx in range(2, self.num_layers):
            z = Zs[-idx]
            prime = select_prime(self.activations[-idx])
            sp = prime(z)
            delta = np.dot(self.weights[-idx + 1].T, delta) * sp
            nabla_b[-idx] = delta
            nabla_w[-idx] = np.dot(delta, As[-idx -1].T)
        return nabla_w, nabla_b

    def update_weights(self, mini_batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        m = len(mini_batch)
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backward_prop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # For each layer adjust the weights accordingly
        self.weights = [w - (eta / m) * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb 
                        for b, nb in zip(self.biases, nabla_b)]

    def train_network(self, training_data, epochs, batch_size, eta, test_data=None):
        if test_data: 
            n_test = len(test_data)
            print(n_test)
        n = len(training_data)
        print(n)
        for j in range(epochs):
            # Shuffle training data
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] 
                            for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_prop(x)), y)
                        for x, y in test_data]
        return sum(int(x==y) for x, y in test_results)

if __name__ == "__main__":
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.import_data()
    layers = [[784, "input"], [32, "sigmoid"], [10, "sigmoid"]]
    net = NeuralNetwork(layers)
    net.train_network(training_data, epochs=5, batch_size=10, eta=3, test_data=test_data)