import numpy as np


def relu(x, derivative = False):
    if derivative:
        return 1 * (x > 0)
    else:
        return x * (x > 0)

'''
Calculates class probabilities, used for multiple classification
'''
def softmax(x, derivative = False):
    if derivative:
        print(f'x: {x.shape}')
        s = x.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)
    else:
        return np.exp(x) / np.exp(x).sum()



'''
Squashes values in the range between 0 and 1, for binary classification
'''
def sigmoid(x, derivative = False):
    if derivative:
        return x * (1.0 - x)
    else:
        return 1 / (1 + np.exp(-x))

def no_activation(x, derivative = False):
    if derivative:
        return 1
    else:
        return x
