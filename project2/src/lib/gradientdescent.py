#!/usr/bin/env python3
import numpy as np

class SGD:

    def __init__(self, learningrate):
        self.learningrate = learningrate

    def betaNew(self, X, y, betaOld):
        """

        iteration of the gradient for beta's pure pseudocode as of now
        takes the current/old beta and returns the new beta.
        This is currently a vector operation, so this would compute the
        entire beta
        """
        #betaNew = betaOld - inverse(X.T@W@X) @ (-X.T@(y-p)) # computed w/ old beta
        
        betaNew = betaOld - self.learningrate*self.gradient(X, y, betaOld)

    def gradient(self, X, y, beta):
        """
        gradient for ols, 
        del(C) = del( (1/n)||X@beta - y||_2^2 = (2/n) X^T @ ( X@beta - y )
        """
        #return (2./y.shape[0])* X.T @ (X@beta - y)
        print("*"*20)
        print("beta")
        print(beta.shape)
        print("X")
        print(X.shape)
        print("*"*20)
        blab = X@beta
        print("*"*20)
        print("blab")
        print(blab.shape)
        print("y")
        print(y.shape)
        print("*"*20)
        blab -= y
        blab = X.T @ blab
        blab *= (2./y.shape[0])
        return blab

    def FindBeta(self,X,y):
        beta = np.random.randn(X.shape[1])#.reshape((X.shape[1],1))
        print("beta, X, y:")
        print(beta.shape)
        print(X.shape)
        print(y.shape)
        print("X@beta")
        print((X@beta).shape)
        print("X.T@y")
        print((X.T@y).shape)

        for i in range(int(1e6)):
            betaOld = beta
            beta = self.betaNew(X,y,betaOld)
        self.betaBest = beta
        return self.betaBest
