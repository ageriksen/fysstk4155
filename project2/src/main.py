#!/usr/bin/env python3

import lib.regressor as reg
import lib.gradientdescent as gd

import systems.simplepoly as smpl
import systems.franke as frnk

import numpy as np
import sklearn.linear_model as skl


def main():
    np.random.seed(2020)
    rows = 100; cols = 50; sigma = 1
    maxdegree = 20; testRatio=.2

    learningrates = np.linspace(0.01, 0.1, 10)
    epochs = np.arange(10, 100)

    franke = frnk.FrankeRegression()
    franke.SetRegressor(gd.SGD)#, epochs=100, minibatches=100)
    franke.SetSklRegressor(skl.LinearRegression)
    franke.SetSystem(rows, cols, sigma) 
    franke.SetLearningMode('static', 0.1)
    franke.Run( testRatio, maxdegree )

if __name__ == "__main__":
    main()
