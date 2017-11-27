#!/bin/python

# Barebones ANN implementation for predicting guilt


import math, copy
from collections import namedtuple
import numpy as np
import parser

INPUT_UNITS = 23
OUTPUT_UNITS = 1
# Hidden layers: 1
HIDDEN_UNITS = INPUT_UNITS
UNITS_IN_LAYER = [23, 23, 1]
LAMBDA = 0.0001
ALPHA = 0.2
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
    return sigmoid(x.dot(theta))

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

def cost(theta, X, y):
    m = len(y)
    cost = 0
    for i in range(m):
        for k in range(OUTPUT_UNITS):
            h_theta_x = calc_unit_outputs(theta, X[i]).output[k]
            cost += y[i][k]*math.log(h_theta_x) + \
                    (1 - y[i][k])*math.log(1 - h_theta_x)

    cost /= -m

    regularizer = 0
    for l in range(len(theta)):
        for i in range(UNITS_IN_LAYER[l]):
            for j in range(UNITS_IN_LAYER[l+1]):
                regularizer += theta[l][j,i]**2

    regularizer *= LAMBDA/(2*m)

    return cost + regularizer

def calc_hidden_deltas(theta, activations, output_deltas):
    deltas = Net.empty()
    deltas = np.matrix([deltas.input, deltas.hidden, output_deltas])
    # Since there is only one hidden layer, just do that one
    return np.multiply(theta[1].T.dot(output_deltas),
                       np.multiply(activations.hidden,
                                   1 - activations.hidden))

# returns the partial derivatives of the cost function wrt THETA
def backprop(theta, X, y):
    L = len(theta)
    m = len(y)

    # Delta and theta have the same shape
    Delta = [np.zeros(UNITS_IN_LAYER[1]*(UNITS_IN_LAYER[0] + 1))\
             .reshape(UNITS_IN_LAYER[1], UNITS_IN_LAYER[0] + 1),
             np.zeros(UNITS_IN_LAYER[2]*(UNITS_IN_LAYER[1] + 1))\
             .reshape(UNITS_IN_LAYER[2], UNITS_IN_LAYER[1] + 1)]
    for i in range(m):
        # forward propagation
        activations = calc_unit_outputs(theta, X[i])

        output_deltas = activations.output - y[i]
        hidden_deltas = calc_hidden_deltas(theta, activations, output_deltas)

        Delta[1] += np.multiply(activations.hidden, output_deltas)
        Delta[0] += np.multiply(activations.input, hidden_deltas)

    D = Delta
    D[0] = D[0]/m + LAMBDA*theta[0]
    D[1] = D[1]/m + LAMBDA*theta[1]

    return D

def gradient_descent(theta, X, y):
    converged = lambda c, last: abs(c - last) < 0.0001

    curr_theta = copy.deepcopy(theta)
    curr_cost = cost(curr_theta, X, y)
    last_cost = 10000

    while not converged(curr_cost, last_cost):
        partials = backprop(curr_theta, X, y)
        temp_theta = copy.deepcopy(curr_theta)

        for l in range(2):
            for i in range(UNITS_IN_LAYER[l + 1]):
                for j in range(UNITS_IN_LAYER[l]):
                    temp_theta[l][i,j] -= ALPHA*partials[l][i,j]


        curr_theta = temp_theta

        last_cost = curr_cost
        curr_cost = cost(curr_theta, X, y)

    return curr_theta

def train(X, y):
    np.random.seed(1)
    theta = [np.random.random((len(X[0]), len(X[0]) + 1)) - 0.5,
             np.random.random((1, len(X[0]) + 1)) - 0.5]
    optimized_theta = gradient_descent(theta, X, y)

    return optimized_theta

def test(theta, X, y):
    correct = 0
    for (i, y_i) in enumerate(y):
        output = calc_unit_outputs(theta, X[i]).output
        if int(round(output)) == y_i:
            correct += 1

    print("training accuracy: {}".format(float(correct)/len(y)))

def main():
    (X_train, y_train) = parser.load_training()
    (X_test, y_test) = parser.load_test()
    theta = train(X_train, y_train)
    test(theta, X_test, y_test)

