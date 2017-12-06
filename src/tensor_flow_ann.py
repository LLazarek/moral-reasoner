#!/bin/python

import tensorflow as tf
import numpy as np
import parser

LEARN_RATE = 0.1

def forward_propogate(hidden_size, X, theta0, theta1):
    biases_hidden = tf.get_variable('biases_hidden', shape=[hidden_size], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
    biases_output = tf.get_variable('biases_output', shape=[2], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
    hidden_activations = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(X, theta0), biases_hidden))
    return tf.nn.bias_add(tf.matmul(hidden_activations, theta1), biases_output)

def train(X_train, y_train, X_test, y_test):
    input_size = X_train.shape[1]
    hidden_size = 10
    y_size = 2

    X = tf.placeholder("float", shape=[None, input_size])
    y = tf.placeholder("float", shape=[None, y_size])


    tf.set_random_seed(101)
    theta0 = tf.Variable(tf.random_normal((X_train.shape[1],
                                           hidden_size),
                                          stddev=0.1))
    theta1 = tf.Variable(tf.random_normal((hidden_size, 2),
                                          stddev=0.1))

    predicted_y = forward_propogate(hidden_size, X, theta0, theta1)
    predict = tf.argmax(predicted_y, axis=1) # Get index of largest value

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=predicted_y))
    updates = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(cost)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    last_train_res = [[0, 1]]*X_train.shape[0]
    last_test_res = [[0, 1]]*X_test.shape[0]

    for iteration in range(100):
        for i in range(X_train.shape[0]):
            #print(X_train[i:i+1], y_train[i:i+1])
            sess.run(updates, feed_dict={X: X_train[i: i + 1], y: y_train[i: i + 1]})

        print(sess.run(theta0))

        train_res = np.matrix(sess.run(predict, feed_dict={X: X_train, y: y_train}))
        train_y_exp = np.matrix(np.argmax(y_train, axis=1)) # Get index of largest value
        print(train_res)
        print(train_y_exp)
        not_same = False
        for i in range(len(train_res)):
            print(train_res[i])
            print(last_train_res[i])
            if train_res[i][0] == last_train_res[i][0]:
                not_same = True
        if not_same:
            print("Training prediction is not changing")
        last_train_res = train_res
        train_accuracy = np.mean(train_y_exp ==
                                 train_res)

        test_res = sess.run(predict, feed_dict={X: X_test, y: y_test})
        not_same = False
        for i in range(len(test_res)):
            if test_res[i][0] == last_test_res[i][0]:
                not_same = True
        if not_same:
            print("Testing prediction is not changing")
        last_test_res = test_res
        test_accuracy = np.mean(np.argmax(y_test, axis=1) ==
                                test_res)

        print("Iteration {}: Train = {}, test = {}".format(iteration, train_accuracy, test_accuracy))

    print(sess.run(predict, feed_dict={X: X_train[15:20], y: y_train[15:20]}))
    print(y_train[15:20])

    sess.close()

def main():
    (X_train, y_train) = parser.load_training()
    # Add bias column
    (m, n) = X_train.shape
    temp = np.ones((m, n + 1))
    temp[:,1:] = X_train
    X_train = temp
    
    (X_test, y_test) = parser.load_test()
    # Add bias column
    (m, n) = X_test.shape
    temp = np.ones((m, n + 1))
    temp[:,1:] = X_test
    X_test = temp

    y_train = tf.Session().run(tf.one_hot(y_train, 2))
    y_train = np.matrix(list(map(lambda x: x[0].tolist(), y_train)))
    y_test = tf.Session().run(tf.one_hot(y_test, 2))
    y_test = np.matrix(list(map(lambda x: x[0].tolist(), y_test)))

    # print(X_test)
    # print(y_test)
    # print(X_train)
    # print(y_train)
    # exit(1)

    train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
