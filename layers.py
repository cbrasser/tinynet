import numpy as np
from activation_function import sigmoid

class HiddenLayer():
    def __init__(self, d_in, d_out):
        print(f'Creating Hidden Layer of size {d_in} X {d_out}')
        self.input_dim = d_in
        self.output_dim = d_out
        # Sample the random weights from a normal distribution
        self.weights = 2* np.random.rand(self.input_dim, self.output_dim) -1
        self.bias = 2* np.random.rand(1, self.output_dim) -1

    def forward(self,x):
        # Forward pass is inputs times weights passed through activation function
        self.output = sigmoid(np.dot(x,self.weights))
        return self.output

class InputLayer():
    def __init__(self, size):
        print(f'creating input layer of size {size}')
        self.size = size

class OutputLayer():
    def __init__(self, d_in, d_out):
        print(f'Creating output layer of size {d_in} X {d_out}')
        self.size = d_in
        self.out_dim = d_out
        self.weights = 2* np.random.rand(self.size, 1) - 1

    def forward(self, x):
        self.output = sigmoid(np.dot(x,self.weights))
        return self.output
