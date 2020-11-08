#!/usr/bin/env python3

import lib.regressor as reg
import lib.gradientdescent as gd

import systems.simplepoly as smpl
import systems.franke as frnk

import numpy as np
import sklearn.linear_model as skl


def main():
    np.random.seed(2020)
    rows = 200; cols = 100; sigma = .5
    maxdegree = 10; testRatio=.2

    #epochs = np.arange(10, 100)
    #minibatches = np.arange(10, 100)#int(rows*cols*.4), int(rows*cols*.8)) 
    #learningrates = np.linspace(0.01, 0.1, 10)
    epochs = np.array([10, 50, 100])
    minibatches = np.array([10, 50, 100])
    #learningrates = np.array([.0001, .001, .01])
    learningrates = np.array([.01])


    franke = frnk.FrankeRegression()
    franke.SetRegressor(gd.SGD)#, epochs=100, minibatches=100)
    franke.SetSklRegressor(skl.LinearRegression)
    franke.SetSystem(rows, cols, sigma) 
    franke.SetLearningMode('static', 0.1)
    franke.Run( testRatio, maxdegree, epochs, minibatches, learningrates)

if __name__ == "__main__":
    main()
