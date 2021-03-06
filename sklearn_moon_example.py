import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=5000, random_state=42, noise=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
y_train = np.asarray([y_train])
y_test = np.asarray([y_test])

print(X_train.shape)

n_epochs = 500
minibatch_size = 10
n_runs = 1

accs = np.zeros(n_runs)

for k in range(n_runs):
    print(f'Run nr - {k+1}')
    nn = NeuralNetwork(2,1, activation = 'sigmoid', learning_rate = 0.1)
    print(f'Training')
    nn.sgd_minibatch(X_train, y_train, minibatch_size, n_epochs)

    predictions = np.zeros(y_test.shape[1])
    print(f'Testing')
    for i, x in enumerate(X_test):
        score = nn.predict(x)
        prediction = 1 if score >=0.5 else 0
        # print(f'pred: {prediction} - true: {y_test[0,i]}')
        predictions[i] = prediction

    accuracy = (predictions == y_test).sum() / y_test.size
    print(f'Accuracy - {accuracy}')
    if n_runs > 1:
        accs[k] = accuracy


if n_runs > 1:
    print(f'Mean accuracy: {accs.mean()} - std: {accs.std()}')
