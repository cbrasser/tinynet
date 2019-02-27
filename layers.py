import numpy as np
import activation_function

class HiddenLayer():
    def __init__(self, d_in, d_out, act):
        print(f'Creating Hidden Layer of size {d_in} X {d_out}')
        self.input_dim = d_in
        self.output_dim = d_out
        self.activation_function = getattr(activation_function,act)
        # Sample the random weights from a normal distribution
        self.weights = np.random.uniform(size=(self.input_dim, self.output_dim))
        self.bias = np.random.uniform(size=(1, self.output_dim))

    def forward(self,x):
        # Forward pass is inputs times weights passed through activation function

        self.output = self.activation_function(np.dot(x,self.weights) + self.bias)
        return self.output

    def backward(self, output, gradients_next_layer, weights_next_layer):
        error = gradients_next_layer.dot(weights_next_layer.T)
        delta = error*self.activation_function(output, derivative =True)
        return error, delta

class InputLayer():
    def __init__(self, size):
        print(f'creating input layer of size {size}')
        self.size = size

class OutputLayer():
    def __init__(self, d_in, d_out, act):
        print(f'Creating output layer of size {d_in} X {d_out}')
        self.size = d_in
        self.out_size = d_out
        self.activation_function = getattr(activation_function,act)
        self.weights = np.random.uniform(size=(self.size, self.out_size))
        self.bias = np.random.uniform(size=(1, self.out_size))

    def forward(self, x):
        self.output = self.activation_function(np.dot(x,self.weights) + self.bias)
        return self.output

    def backward(self, output, y):
        error = y.T - output
        delta = error * self.activation_function(output, derivative = True)
        return error, delta
