from layers import HiddenLayer, InputLayer, OutputLayer
import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, output_size,num_hidden = 1, hidden_layer_size = 6, learning_rate = 0.01):
        self.input_layer = InputLayer(input_size)
        # self.hidden_layer_1 = HiddenLayer(input_size,hidden_layer_size, 'relu')
        self.hidden_layer_2 = HiddenLayer(input_size,hidden_layer_size, 'relu')
        self.output_layer = OutputLayer(hidden_layer_size, output_size, 'sigmoid')
        self.lr = learning_rate


    # Forward pass through the neural network
    def forward(self, X):
        # self.z = self.hidden_layer_1.forward(X)
        self.z2 = self.hidden_layer_2.forward(X) #z2
        return self.output_layer.forward(self.z2) # output

    '''
    Backpropagation to find the partial derivatives by applying the chain-rule. Needs training
    samples X, corresponding gold truth labels y, and the predictions from the forward pass (output).
    Adjusts the weights directly according to the derivatives.
    '''
    def backprop(self,X,y,output):

        self.output_error, self.output_delta = self.output_layer.backward(output, y)

        self.z2_error, self.z2_delta = self.hidden_layer_2.backward(self.z2, self.output_delta, self.output_layer.weights)

        # self.z_error, self.z_delta = self.hidden_layer_1.backward(self.z, self.z2_delta, self.hidden_layer_2.weights)

        # self.hidden_layer_1.weights += self.lr * X.T.dot(self.z_delta)
        # self.hidden_layer_1.bias += self.lr * np.sum(self.z_delta, axis=0,keepdims=True)
        self.hidden_layer_2.weights += self.lr * X.T.dot(self.z2_delta)
        self.hidden_layer_2.bias += self.lr * np.sum(self.z2_delta, axis=0,keepdims=True)
        self.output_layer.weights += self.lr * self.z2.T.dot(self.output_delta)
        self.output_layer.bias += self.lr * np.sum(self.output_delta, axis=0,keepdims=True)

    # One cycle of applying forward and backward passes
    def train(self, X,y):
        output = self.forward(X)
        self.backprop(X,y,output)


    def sgd(self, X_train, y_train, minibatch_size, n_epochs):
        for epoch in range(n_epochs):
            # print('Iteration {}'.format(iter))

            # Randomize data point
            np.random.shuffle(X_train)
            np.random.shuffle(y_train)

            for i in range(0, X_train.shape[0], minibatch_size):
                # Get pair of (X, y) of the current minibatch/chunk
                X_train_mini = X_train[i:i + minibatch_size]
                y_train_mini = np.asarray([y_train[0,i:i + minibatch_size]])
                self.train(X_train_mini, y_train_mini)
