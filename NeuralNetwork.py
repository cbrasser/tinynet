from layers import HiddenLayer, InputLayer, OutputLayer
import numpy as np
from sklearn.utils import shuffle

class NeuralNetwork():
    def __init__(self, input_size, output_size, activation_hidden, activation_out, num_hidden = 1, hidden_layer_size = 3, learning_rate = 0.01):
        self.input_layer = InputLayer(input_size)
        self.hidden_layer = HiddenLayer(input_size,hidden_layer_size, activation_hidden)
        self.output_layer = OutputLayer(hidden_layer_size, output_size, activation_out)
        self.lr = learning_rate


    # Forward pass through the neural network
    def forward(self, X):
        self.z = self.hidden_layer.forward(X)
        self.output = self.output_layer.forward(self.z)

    def predict(self, X):
        h = self.hidden_layer.forward(X)
        return self.output_layer.forward(h)

    '''
    Backpropagation to find the partial derivatives by applying the chain-rule. Needs training
    samples X, corresponding gold truth labels y, and the predictions from the forward pass (output).
    Adjusts the weights directly according to the derivatives.
    '''
    def backprop(self,X,y):

        self.output_delta = self.output_layer.backward(self.output, y)

        self.z_delta = self.hidden_layer.backward(self.z, self.output_delta, self.output_layer.weights)
        self.hidden_layer.weights += self.lr * X.T.dot(self.z_delta)
        self.output_layer.weights += self.lr * self.z.T.dot(self.output_delta)

        self.hidden_layer.bias += self.lr * np.sum(self.z_delta, axis=0,keepdims=True)
        self.output_layer.bias += self.lr * np.sum(self.output_delta, axis=0,keepdims=True)


    def sgd_minibatch(self, X_train, y_train, minibatch_size, n_epochs):
        for epoch in range(n_epochs):
            print(f'------------------epoch {epoch} ---------------------')
            if y_train.shape[0]  == 1:
                y_train = y_train.T
            X_train, y_train = shuffle(X_train, y_train)

            for i in range(0, X_train.shape[0], minibatch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                X_train_mini = X_train[i:i + minibatch_size]
                y_train_mini = np.asarray([y_train[i:i + minibatch_size,0]])

                self.forward(X_train_mini)
                self.backprop(X_train_mini,y_train_mini.T)

    def sgd(self, X_train, y_train, n_epochs):

        X_train, y_train = shuffle(X_train, y_train)

        for epoch in range(n_epochs):
            # lmao wtf was i thinking hahahaha, literally cost me 2 hours
            # np.random.shuffle(X_train)
            # np.random.shuffle(y_train)

            self.forward(X_train)
            self.backprop(X_train,y_train)
