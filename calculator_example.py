import pandas as pd
import numpy as np
import inflect

from NeuralNetwork import NeuralNetwork
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

p = inflect.engine()


def int_to_word(number):
    return p.number_to_words(number)

def to_bigram(string):
    temp = str.split(string, ' ')
    new = []
    for i in range(len(temp)-1):
        new.append([temp[i],temp[i+1]])
    return new

def training_data(size):
    x = []
    y = []

    for i in range(size):
        a = np.random.randint(999)
        b = np.random.randint(999)
        c = '+'
        if np.random.random(1) > 0.5:
            c = '-'
        x.append([a, b, ord(c)])
        y.append(a + b if c == '+' else a-b)

    return x, y

def test_data(size):
    x = []
    y = []

    for i in range(size):
        a = np.random.randint(999)
        b = np.random.randint(999)
        c = 1
        if np.random.random(1) > 0.5:
            c = -1
        x.append([a, b, c])
        y.append(a + b if c == 1 else a-b)

    return x, y

x,y = training_data(10000)

X_train = np.asarray(x)
y_train = np.asarray([y])

x,y = test_data(500)

X_test = np.asarray(x)
y_test = np.asarray([y])


n_epochs = 500
minibatch_size = 10
n_runs = 1

accs = np.zeros(n_runs)

for k in range(n_runs):
    print(f'Run nr - {k+1}')
    nn = NeuralNetwork(3,1, activation_hidden = 'sigmoid',activation_out='no_activation', learning_rate = 0.1)
    print(f'Training')
    nn.sgd_minibatch(X_train, y_train, minibatch_size, n_epochs)

    predictions = np.zeros(y_test.shape[1])
    print(f'Testing')
    for i, x in enumerate(X_test):
        score = nn.predict(x)
        print(f'task: {x} - prediction: {score}')
        prediction = 1 if score >=0.5 else 0
        # print(f'pred: {prediction} - true: {y_test[0,i]}')
        predictions[i] = prediction

    accuracy = (predictions == y_test).sum() / y_test.size
    print(f'Accuracy - {accuracy}')
    if n_runs > 1:
        accs[k] = accuracy


if n_runs > 1:
    print(f'Mean accuracy: {accs.mean()} - std: {accs.std()}')
