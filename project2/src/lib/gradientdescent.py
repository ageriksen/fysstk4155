#!/usr/bin/env python3
from tqdm import tqdm
import numpy as np

class _gradientDescent:
    def __init__(self, learningrate):
        self.learningrate = learningrate
        self.null = False

    def gradient(self, X, y, beta):
        """
        gradient for ols, 
        del(C) = del( (1/n)||X@beta - y||_2^2 = (2/n) X^T @ ( X@beta - y )
        """
        gradient = (2./y.shape[0])* X.T @ (X@beta - y)
        if np.linalg.norm(gradient) <= 1e-8:
            self.null = True
        return gradient

class SGD(_gradientDescent):
    def __init__(self, *args):
        super().__init__(*args)
        self.epochs = 50 #need to configure setting epochs.
        self.t0, self.t1 = 5, 50 #for varyin learningrate as we go. need to implement setting.

    #def next(self, X, y, old):
    #    return old - self.learningrate*self.gradient(X,y,old)

    def lrnschdl(self, t): return self.t0 / (t + self.t1)

    def FindBeta(self,X,y, maxiter=1e3):
        beta = np.zeros(X.shape[1])#.reshape((X.shape[1],1))
        self.null = False
        theta = np.random.randn(X.shape[1], 1)
        for epoch in tqdm(range(self.epochs)):
            for i in range(y.shape[0]):
                rand= np.random.randint(y.shape[0])
                Xi = X[rand:rand+1]
                yi = y[rand:rand+1]
                gradients = self.gradient(Xi, yi, theta) 
                eta = self.lrnschdl(epoch*y.shape[0]+1) #the y.shape here is the minibatch and should really be declared at some point
                theta -= eta*gradients
            if self.null:
                self.betaBest = theta
        self.betaBest = theta
        return self.betaBest
        
class GD(_gradientDescent):

    def __init__(self, *args):
        super().__init__(*args)

    def betaNew(self, X, y, betaOld):
        return betaOld - self.learningrate*self.gradient(X, y, betaOld)

    def FindBeta(self,X,y, maxiter=1e3):
        beta = np.zeros(X.shape[1])#.reshape((X.shape[1],1))
        self.null = False
        for i in tqdm(range(int(maxiter))):
            betaOld = beta
            beta = self.betaNew(X,y,betaOld)
            if self.null:
                self.betaBest = beta
                return self.betaBest
        self.betaBest = beta
        return self.betaBest
