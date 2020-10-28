#!/usr/bin/env python3

import lib.regressor as reg
import lib.gradientdescent as gd

import systems.simplepoly as smpl
import systems.franke as frnk

import numpy as np


def main():
    np.random.seed(2020)
    rows = 100; cols = 100; sigma = 0.1
    maxdegree = 10; testRatio=.2
    #RunFranke(rows, cols, sigma)
    franke = frnk.FrankeRegression(gd.SGD, epochs=50, minibatches=10)
    franke.Set(rows, cols, sigma) 
    franke.Run(testRatio, maxdegree)
    #franke.RunFranke(rows, cols, sigma)

    #dtpts = 100
    #simple = smpl.SimpleRegression()
    #simple.RunSimpleExample(dtpts)


    

if __name__ == "__main__":
    main()
