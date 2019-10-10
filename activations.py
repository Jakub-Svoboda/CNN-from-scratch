import math
import numpy as np


def relu(inputsSum):	
    if inputsSum < 0:
        return inputsSum * 0.001
    else:
        return inputsSum	

def softMax(vector):
    idx = vector.index(max(vector))
    for idx2, _ in enumerate(vector):
        if idx == idx2:
            vector[idx2] = 1
        else:
            vector[idx2] = 0	

def linear(input):
    return input


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)
