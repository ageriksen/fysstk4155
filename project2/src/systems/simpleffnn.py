#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import nn.ff as ffnn
import lib.activationfunc as active
import lib.gradientdescent as gd

np.random.seed(2020)

def simpleNN():

    n = 100
    x = np.linspace(0,1,n).reshape(1,-1)
    y = 1.1 - 3*x + 3*x**2 + np.random.normal(0, 0.1, (1,n)) < 0.5

    plt.plot(x[y==0], y[y==0], "bo")
    plt.plot(x[y==1], y[y==1], "ro")
    plt.show()

    X = np.vstack( (x, x**2) )

    dim = [2, 5, 1]

    nn = ffnn.FFNN(active.Sigmoid)
    weights, biases = nn.MakeWeightsAndBiases(dim)

    y_tilde, _, _ = nn.FF(X, weights, biases)
    y_tilde = np.round(y_tilde)

    plt.plot(x[y_tilde==0], y_tilde[y_tilde==0], "bo")
    plt.plot(x[y_tilde==1], y_tilde[y_tilde==1], "ro")
    plt.show()

    eta = 1 # learningrate
    epochs = int(1e3)
    for epoch in range(epochs):
        grad_weights, grad_biases = nn.BackPropagate(X, y, weights, biases)
        for i in range(len(weights)):
            weights[i] -= eta*grad_weights[i]
            biases[i] -= eta*grad_biases[i]

    y_tilde, _, _ = nn.FF(X, weights, biases)
    y_tilde = np.round(y_tilde)

    plt.plot(x[y_tilde==0], y_tilde[y_tilde==0], "bo")
    plt.plot(x[y_tilde==1], y_tilde[y_tilde==1], "ro")
    plt.show()
