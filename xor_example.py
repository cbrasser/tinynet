import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def 

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([ [0],   [1],   [1],   [0]])

n_epochs = 60000
minibatch_size = 4
n_runs = 1

accs = np.zeros(n_runs)

for k in range(n_runs):
    print(f'Run nr - {k+1}')
    nn = NeuralNetwork(2,1, 'sigmoid', learning_rate = 1)
    print(f'Training')
    nn.sgd(X, y, n_epochs)

    predictions = np.zeros(y.shape[0])
    print(f'result')
    print(nn.output)
