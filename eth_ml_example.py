import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_sample = pd.read_csv('data/sample.csv')

# Training data
df_train_X = df_train.loc[:,'x1':'x10']
df_train_Y = df_train.loc[:,'y']

# Testing data
df_test_X = df_test.loc[:,'x1':'x10']
df_test_Y = df_sample.loc[:,'y']

# Convert to numpy arrays, because fast
X_train = np.asarray(df_train_X)
y_train = np.asarray([df_train_Y])

X_test = np.asarray(df_test_X)
y_test = np.asarray([df_test_Y])

n_epochs = 500
minibatch_size = 10
n_runs = 1

accs = np.zeros(n_runs)

for k in range(n_runs):
    print(f'Run nr - {k+1}')
    nn = NeuralNetwork(10,1, activation = 'relu', learning_rate = 0.1)
    print(f'Training')
    nn.sgd_minibatch(X_train, y_train, minibatch_size, n_epochs)

    predictions = np.zeros(y_test.shape[1])
    print(f'Testing')
    for i, x in enumerate(X_test):
        score = nn.predict(x)
        print(f'prediction: {score} - true: {y_test[0,i]}')
        # print(f'pred: {prediction} - true: {y_test[0,i]}')
        predictions[i] = score

    accuracy = (np.abs(predictions - y_test)).sum() / y_test.size
    print(f'Accuracy - {accuracy}')
    if n_runs > 1:
        accs[k] = accuracy


if n_runs > 1:
    print(f'Mean accuracy: {accs.mean()} - std: {accs.std()}')
