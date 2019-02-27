import numpy as np



def relu():
    pass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)
