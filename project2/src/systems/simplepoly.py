#!/usr/bin/env python3
import lib.regressor as reg
import lib.gradientdescent as gd

import numpy as np
import matplotlib.pyplot as plt

class SimpleRegression:

    def __init__(self):
        self

    def RunSimpleExample(self, dtpts):
        from sklearn.linear_model import SGDRegressor
        x = np.random.rand(dtpts,1) # random col matrix of positions
        y = 4 + 3*x*np.random.randn(dtpts,1) # polynomial with noise 

        X = np.c_[np.ones((dtpts, 1)), x] # feature matrix w bias and inputs
        
        print("="*50, "\nBeta's:")

        #Theta is the common nomenclature for optimization params. 
        theta_ols = np.linalg.pinv( X.T@X ) @ ( X.T@y )
        print("ols theta\n", theta_ols)

        #skl
        sklreg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
        sklreg.fit(x, y.ravel())
        print("skl regr.\n", sklreg.intercept_, sklreg.coef_)

        #GD
        lrnrt = 0.01
        maxiter = 1e5
        grad = gd.GD(lrnrt)
        GDBeta = grad.FindBeta(X,y.ravel(), maxiter)
        print("own GD\n", GDBeta)

        #SGD
        sgrad = gd.SGD(lrnrt)
        SGDBeta = sgrad.FindBeta(X, y.ravel(), maxiter)
        print("own SGD\n", SGDBeta)

        print("="*50)
        
        plt.figure()
        plt.plot(x, y, label="y")
        plt.plot(x, X@theta_ols, label="ols fit")
        plt.plot(x, X@GDBeta, label="GD fit")
        plt.plot(x, X@SGDBeta, label="SGD fit")
        plt.legend()
        plt.show()
