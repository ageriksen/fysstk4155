#!/usr/bin/env python3

import lib.regressor as reg
import lib.gradientdescent as gd

import systems.simplepoly as smpl
import systems.franke as frnk

import numpy as np


def main():
    np.random.seed(2020)
    rows = 100; cols = 100; sigma = 1
    maxdegree = 15; testRatio=.2

    franke = frnk.FrankeRegression(gd.SGD, epochs=500, minibatches=100)
    franke.Set(rows, cols, sigma) 
    franke.Run( testRatio, maxdegree )


    

if __name__ == "__main__":
    main()
