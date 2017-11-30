#!/bin/python

import tensorflow as tf
import numpy as np
import parser


def forward_propogate(theta, X):
    hidden_activations = tf.nn.sigmoid(tf.matmul(X, theta[0]))
    return tf.matmul(hidden_activations, theta[1])

def main():
    (X_train, y_train) = parser.load_training()
    # Add bias column
    (m, n) = X_train.shape
    temp = np.ones(m, n + 1)
    temp[:,1:] = X_train
    X_train = temp
    
    (X_test, y_test) = parser.load_test()
    # Add bias column
    (m, n) = X_test.shape
    temp = np.ones(m, n + 1)
    temp[:,1:] = X_test
    X_test = temp

    input_size = X_train.shape[1]
    hidden_size = input_size
    y_size = 1

    X = tf.placeholder("float", shape=[None, input_size])
    y = tf.placeholder("float", shape=[None, y_size])


    tf.set_random_seed(1)
    theta = [np.random.random((len(X[0]), len(X[0]) + 1)) - 0.5,
             np.random.random((1,         len(X[0]) + 1)) - 0.5]

    predicted_y = forward_propogate(theta, X)
    predict = tf.argmax(y, axis=1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=predicted_y))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for iteration in range(100):
        for i in range(len(train_x)):
            sess.run(updates, feed_dict={X: X_train[i: i + 1], y: y_train[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: X_train, y: y_train}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: X_test, y: y_test}))

        print("Iteration {}: Train = {}, test = {}".format(iteration, train_accuracy, test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()
