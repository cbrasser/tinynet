from layers import HiddenLayer, InputLayer, OutputLayer
from activation_function import sigmoid_derivative
import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, output_size,num_hidden = 1, hidden_layer_size = 6):
        self.input_layer = InputLayer(input_size)
        self.hidden_layer_1 = HiddenLayer(input_size,hidden_layer_size)
        self.hidden_layer_2 = HiddenLayer(hidden_layer_size,hidden_layer_size)
        self.output_layer = OutputLayer(hidden_layer_size, output_size)


    # Forward pass through the neural network
    def forward(self, X):
        self.z = self.hidden_layer_1.forward(X)
        self.z2 = self.hidden_layer_2.forward(self.z) #z2
        return self.output_layer.forward(self.z2) # output

    '''
    Backpropagation to find the partial derivatives by applying the chain-rule. Needs training
    samples X, corresponding gold truth labels y, and the predictions from the forward pass (output).
    Adjusts the weights directly according to the derivatives.
    '''
    def backprop(self,X,y,output):

        self.output_error = y.T - output
        self.output_delta = self.output_error * sigmoid_derivative(output)

        self.z2_error = self.output_delta.dot(self.output_layer.weights.T)
        self.z2_delta = self.z2_error*sigmoid_derivative(self.z2)

        self.z_error = self.z2_delta.dot(self.hidden_layer_2.weights.T)
        self.z_delta = self.z_error*sigmoid_derivative(self.z)

        self.hidden_layer_1.weights += 0.01 * X.T.dot(self.z_delta)
        self.hidden_layer_2.weights += 0.01 * self.z.T.dot(self.z2_delta)
        self.output_layer.weights += 0.01 * self.z.T.dot(self.output_delta)

    # One cycle of applying forward and backward passes
    def train(self, X,y):
        output = self.forward(X)
        self.backprop(X,y,output)
