#!/bin/python

# Barebones ANN implementation for predicting guilt


import math, copy
from collections import namedtuple
import numpy as np
import parser
from datetime import datetime

CHECK_GRADIENT = False
# Hidden layers: 1
UNITS_IN_LAYER = [23, 23, 1]
LAMBDA = 0.001#001
ALPHA = 0.1
class Net(namedtuple('Net', ['input', 'hidden', 'output'])):
    @staticmethod
    def empty(input=np.zeros((1, UNITS_IN_LAYER[0]))):
        input_with_bias = np.append([[1]], input)
        hidden_with_bias = np.append([[1]], np.zeros((1, UNITS_IN_LAYER[1])))
        return Net(input=input_with_bias,
                   hidden=hidden_with_bias,
                   output=np.zeros((1, UNITS_IN_LAYER[-1])))


def sigmoid_value(x):
    return 1.0/(1.0 + math.exp(-x))

sigmoid_matrix = np.vectorize(sigmoid_value)

def sigmoid(x):
    return sigmoid_matrix(x) if isinstance(x, np.matrix) \
        else sigmoid_value(x)

# theta: (m x n+1)
# x: (1 x n+1)
def h(theta, x): # h: (m x 1)
    return sigmoid(x.dot(theta))

# calc_unit_outputs: Theta Sample -> Net
# Implements forward propogation to calculate the activations of every unit
# in the net
# x: (1 x 23), theta[0]: (23 x 24)
def calc_unit_outputs(theta, x): # Verified
    assert(len(x) == UNITS_IN_LAYER[0])
    x_with_bias = np.matrix(np.append([1], x)).T
    # theta[0]: (23x24)
    # x_with_bias: (24 x 1)
    hidden = sigmoid(theta[0]*x_with_bias)
    # hidden: (23x1)
    # theta[1]: (1 x 24)
    hidden_with_bias = np.matrix(np.append([[1]], hidden)).T
    # hidden_with_bias: (24x1)
    output = sigmoid(theta[1]*hidden_with_bias)
    return Net(input=x_with_bias, hidden=hidden_with_bias, output=output)

def cost(theta, X, y): # Verified
    m = len(y)
    cost = 0
    for i in range(m):
        for k in range(UNITS_IN_LAYER[-1]):
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
    # g_prime: (24 x 1)
    # theta[1]: (1 x 24)
    # output_deltas: (1 x 1)
    return np.multiply(theta[1].T.dot(output_deltas),
                       g_prime)

# returns the partial derivatives of the cost function wrt THETA
def backprop(theta, X, y):
    L = len(theta)
    m = len(y)

    # Delta and theta have the same shape
    # theta[0]: (23x24)
    # theta[1]: (1 x 24)
    Delta = [np.zeros(theta[0].shape), np.zeros(theta[1].shape)]

    for i in range(m):
        # forward propagation
        activations = calc_unit_outputs(theta, X[i])
        # activations.input: (24 x 1)
        # activations.hidden: (24 x 1)
        # activations.output: (1 x 1)

        # print("activations: {}".format(activations))

        output_deltas = activations.output - y[i]
        output_deltas = np.multiply(output_deltas,
                                    np.multiply(activations.output,
                                                1 - activations.output))
        output_deltas = np.matrix(output_deltas)
        # output_deltas: (1 x 1)

        # print("o delta: {}".format(output_deltas))

        hidden_deltas = calc_hidden_deltas(theta, activations, output_deltas)
        # hidden_deltas: (24 x 1)

        # print("hidden deltas: {}".format(hidden_deltas))

        Delta[1] += output_deltas.dot(activations.hidden.T)

        # I do this because the hidden layer's bias node's delta
        # doesn't propagate back to the input layer; Because the hidden
        # layer's bias node doesn't have any connections to the input
        # layer
        hidden_deltas_minus_bias = hidden_deltas[1:,:] # Xinzi: don't do this
        # hidden_deltas_minus_bias: (23 x 1)
        Delta[0] += hidden_deltas_minus_bias.dot(activations.input.T)
        # print(Delta)

    D = Delta

    # make lambda multiply to 0 for bias weights
    tmp_theta_0 = copy.deepcopy(theta[0])
    tmp_theta_0[:,1] = 0
    tmp_theta_1 = copy.deepcopy(theta[1])
    tmp_theta_1[:,1] = 0

    D[0] = D[0]/m + LAMBDA*tmp_theta_0
    D[1] = D[1]/m + LAMBDA*tmp_theta_1

    return D

def gradient_descent(theta, X, y):
    converged = lambda c, last: abs(c - last) < 0.000001

    curr_theta = copy.deepcopy(theta)
    curr_cost = cost(curr_theta, X, y)
    last_cost = 10000

    count = 0
    REPORT_FREQUENCY = 1000

    while not converged(curr_cost, last_cost):
        if not curr_cost < last_cost:
            print("ERROR: cost not going down")
            exit(1)

        count = (count + 1)%REPORT_FREQUENCY
        if count == 0:
            print("{}: Iteration".format(datetime.now()))
            print("Cost: {}\n".format(curr_cost))
        # print("DOING BACKPROP")
        partials = backprop(curr_theta, X, y)
        # print(partials)

        if CHECK_GRADIENT:
            gradient_check(partials, curr_theta, X, y)
            print("EXITING...\n\n")
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
    epsilon = 0.0000000001
    approx_eq = lambda a, b: abs(a - b) < 0.0001
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
    theta = [np.matrix(np.random.random((len(X[0]), len(X[0]) + 1))) - 0.5,
             np.matrix(np.random.random((1,         len(X[0]) + 1))) - 0.5]
    optimized_theta = gradient_descent(theta, X, y)

    return optimized_theta

def test(theta, X, y, name):
    correct = 0
    for (i, y_i) in enumerate(y):
        output = calc_unit_outputs(theta, X[i]).output
        if int(round(output)) == y_i:
            correct += 1

    print("{} accuracy: {}".format(name, float(correct)/len(y)))

def main():
    print("Loading data...")
    (X_train, y_train) = parser.load_training()
    (X_test, y_test) = parser.load_test()
    print("Training...")
    theta = train(X_train, y_train)
    test(theta, X_train, y_train, "training")
    test(theta, X_test, y_test, "test")
