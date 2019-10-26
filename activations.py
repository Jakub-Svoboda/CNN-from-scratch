#SFC project - CNN from scratch
#Author: Jakub Svoboda 
#Date: 2019-10-26

import math
import numpy as np
import warnings

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

def sigmoidDerivative(x):
	return x * (1.0 - x)

def ReLU(x):
	return np.maximum(x, 0.)

def ReLUDerivative(x):
	x[x > 0.] = 1.
	x[x <= 0.] = 0.
	return x