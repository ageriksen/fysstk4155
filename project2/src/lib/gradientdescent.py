#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm

class _gradientDescent:
    def __init__(self):
        self.null = False

    def gradient(self, X, y, beta):
        """
        gradient for OLS, 
        del(C) = del( (1/n)||X@beta - y||_2^2 = (2/n) X^T @ ( X@beta - y )
        """
        gradient = (2./y.shape[0])* X.T @ (X@beta - y)
        if np.linalg.norm(gradient) <= 1e-5:
            self.null = True
        return gradient

class SGD(_gradientDescent):
    def __init__(self, minibatches, epochs, *args):
        super().__init__(*args)
        self.epochs = epochs #need to configure setting epochs.
        self.t0, self.t1 = 5, 50 #for varying learningrate as we go.
                                #to implement setting variable
        self.minibatches = minibatches

    def lrnschdl(self, t): return self.t0 / (t + self.t1)

    def fit(self, X, y):
        self.null = False
        batchsize = int(y.shape[0]/self.minibatches)
        theta = np.random.randn(X.shape[1], batchsize)
        for epoch in range(self.epochs):
            for i in range(self.minibatches):
                rand = np.random.randint(self.minibatches)
                batch = rand*batchsize
                Xi = X[batch:batch+batchsize]
                yi = y[batch:batch+batchsize]
                gradients = self.gradient(Xi, yi, theta) 
                eta = self.lrnschdl(epoch*y.shape[0]+1) #the y.shape here is the minibatch and should really be declared at some point
                theta -= eta*gradients
            if self.null:
                print("found a 0")
                self.theta = theta
                return self.theta
        self.theta = theta
        #return self.theta
        
class GD(_gradientDescent):

    def __init__(self, learningrate, *args):
        self.learningrate = learningrate
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
                self.theta = beta
                return self.theta
        self.theta = beta
        return self.theta
