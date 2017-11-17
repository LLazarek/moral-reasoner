#!/bin/python

# Barebones ANN implementation for predicting guilt

# Input units: 23
# Output units: 1
# Hidden layers: 1
# Hidden unts/layer: 23

import math
import numpy as np

def sigmoid_value(x):
    return 1.0/(1.0 + math.exp(-x))

sigmoid_matrix = np.vectorize(sigmoid_value)

def sigmoid(x):
    return sigmoid_matrix(x) if isinstance(x, np.matrix) \
        else sigmoid_value(x)

# theta: n x 1
# x: m x n
def h(theta, x):
    return sigmoid(x.dot(theta))

# Vector[n] Matrix[1xn] -> List[List[input]]
def calc_unit_outputs(theta, x):
    
