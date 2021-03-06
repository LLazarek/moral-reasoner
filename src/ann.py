#!/bin/python

# Barebones ANN implementation for predicting guilt


import math, copy
from collections import namedtuple
import numpy as np
import parser

CHECK_GRADIENT = True
INPUT_UNITS = 3
OUTPUT_UNITS = 1
# Hidden layers: 1
HIDDEN_UNITS = INPUT_UNITS
UNITS_IN_LAYER = [3, 3, 1]
LAMBDA = 0.2
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
def calc_unit_outputs(theta, x): # Verified
    assert(x.size == INPUT_UNITS)
    x_with_bias = np.append([1], x)
    filled = Net.empty(input=x)
    for h_i in range(1, filled.hidden.size):
        filled.hidden[h_i] = h(theta[0][h_i - 1,:].T, filled.input)

    for o_i in range(filled.output.size):
        filled.output[o_i] = h(theta[1][o_i,:].T, filled.hidden)

    return filled

def cost(theta, X, y): # Verified
    m = len(y)
    cost = 0
    for i in range(m):
        for k in range(OUTPUT_UNITS):
            h_theta_x = calc_unit_outputs(theta, X[i]).output[k]
            cost += y[i][k]*math.log(h_theta_x) + \
                    (1 - y[i][k])*math.log(1 - h_theta_x)

    cost /= -m

    regularizer = 0
    for layer_theta in theta:
        squared = np.multiply(layer_theta, layer_theta)
        regularizer += np.sum(squared)

    regularizer *= LAMBDA/(2*m)

    return cost + regularizer

def calc_hidden_deltas(theta, activations, output_deltas): # Verified
    # Since there is only one hidden layer, just do that one
    g_prime = np.multiply(activations.hidden, 1 - activations.hidden)
    g_prime[0] = 1 # Otherwise bias always goes to 0
    return np.multiply(theta[1].T.dot(output_deltas),
                       g_prime)

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
        print("activations: {}".format(activations))

        output_deltas = activations.output - y[i]
        output_deltas = np.multiply(output_deltas,
                                    np.multiply(activations.output,
                                                1 - activations.output)) # Verif
        print("o delta: {}".format(output_deltas))
        
        hidden_deltas = calc_hidden_deltas(theta, activations, output_deltas)
        print("hidden deltas: {}".format(hidden_deltas))

        gradient_wrt_weight_1 = np.multiply(activations.hidden, output_deltas)
        gradient_wrt_weight_1[0] = output_deltas[0]
        Delta[1] += gradient_wrt_weight_1

        gradient_wrt_weight_2 = np.multiply(activations.input, hidden_deltas)
        gradient_wrt_weight_2[0] = hidden_deltas[0]
        Delta[0] += gradient_wrt_weight_2

    D = Delta
    tmp_theta_0 = theta[0]
    tmp_theta_0[:,1] = 0
    tmp_theta_1 = theta[1]
    tmp_theta_1[:,1] = 0
    
    D[0] = D[0]/m + LAMBDA*tmp_theta_0
    D[1] = D[1]/m + LAMBDA*tmp_theta_1

    return D

def gradient_descent(theta, X, y):
    converged = lambda c, last: abs(c - last) < 0.0000001

    curr_theta = copy.deepcopy(theta)
    curr_cost = cost(curr_theta, X, y)
    last_cost = 10000

    while not converged(curr_cost, last_cost):
        partials = backprop(curr_theta, X, y)

        if CHECK_GRADIENT:
            gradient_check(partials, curr_theta, X, y)
            exit(1)

        temp_theta = copy.deepcopy(curr_theta)

        for l in range(2):
            for i in range(UNITS_IN_LAYER[l + 1]):
                for j in range(UNITS_IN_LAYER[l]):
                    temp_theta[l][i,j] -= ALPHA*partials[l][i,j]


        curr_theta = temp_theta

        last_cost = curr_cost
        curr_cost = cost(curr_theta, X, y)

    return curr_theta

def gradient_check(partials, theta, X, y):
    epsilon = 0.0001
    approx_eq = lambda a, b: abs(a - b) < 0.000001
    for l in range(len(theta)):
        (rows, cols) = theta[l].shape
        for i in range(rows):
            for j in range(cols):
                temp_theta_plus_e = copy.deepcopy(theta)
                temp_theta_plus_e[l][i,j] += epsilon
                cost_plus_e = cost(temp_theta_plus_e, X, y)
                temp_theta_minus_e = copy.deepcopy(theta)
                temp_theta_minus_e[l][i,j] -= epsilon
                cost_minus_e = cost(temp_theta_minus_e, X, y)
                gradient = (cost_plus_e - cost_minus_e)/(2*epsilon)
                partial = partials[l][i,j]
                if not approx_eq(partial, gradient):
                    print("Gradient check failed! for layer {}, weight {},{}; "
                          "partial {} !~= {}"\
                          .format(l, i, j, partial, gradient))

def train(X, y):
    np.random.seed(1)
    theta = [np.random.random((len(X[0]), len(X[0]) + 1)) - 0.5,
             np.random.random((1,         len(X[0]) + 1)) - 0.5]
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
    print("Loading data...")
    (X_train, y_train) = parser.load_training()
    (X_test, y_test) = parser.load_test()
    print("Training...")
    theta = train(X_train, y_train)
    test(theta, X_test, y_test)
