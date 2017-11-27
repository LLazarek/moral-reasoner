#!/bin/python

# Barebones ANN implementation for predicting guilt


import math, copy
from collections import namedtuple
import numpy as np

INPUT_UNITS = 2
OUTPUT_UNITS = 1
# Hidden layers: 1
HIDDEN_UNITS = INPUT_UNITS
class Net(namedtuple('Net', ['input', 'hidden', 'output'])):
    @staticmethod
    def empty(input=np.zeros(INPUT_UNITS)):
        input_with_bias = np.append([1], input)
        hidden_with_bias = np.append([1], np.zeros(HIDDEN_UNITS))
        return Net(input=input_with_bias,
                   hidden=hidden_with_bias,
                   output=np.zeros(OUTPUT_UNITS))


def sigmoid_value(x):
    return 1.0/(1.0 + math.exp(-x))

sigmoid_matrix = np.vectorize(sigmoid_value)

def sigmoid(x):
    return sigmoid_matrix(x) if isinstance(x, np.matrix) \
        else sigmoid_value(x)

# theta: n x 1
# x: m x n
def h(theta, x):
    # return sigmoid(x.dot(theta))
    print(theta)
    print(x)
    return x.dot(theta)

# calc_unit_outputs: Theta Sample -> Net
# Theta: List[np.matrix[2x3]]
# Implements forward propogation to calculate the activations of every unit
# in the net
def calc_unit_outputs(theta, x):
    assert(x.size == INPUT_UNITS)
    x_with_bias = np.append([1], x)
    filled = Net.empty(input=x)
    for h_i in range(1, filled.hidden.size):
        filled.hidden[h_i] = h(theta[0][h_i - 1,:].T, filled.input)
        # for i in range(filled.input.size):
        #     filled.hidden[h] += filled.input[i]*theta[i]
    for o_i in range(filled.output.size):
        filled.output[o_i] = h(theta[1][o_i - 1,:].T, filled.hidden)

    return filled


