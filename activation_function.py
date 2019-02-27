import numpy as np


'''
TODO
'''
def relu():
    pass

'''
Squashes values in the range between 0 and 1
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)
