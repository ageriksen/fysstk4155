#!/usr/bin/env python3
from tqdm import tqdm
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
        #betaNew = betaOld - inverse(X.T@W@X) @ (-X.T@(y-p)) # computed w/ old beta
        """
        betaNew = betaOld - self.learningrate*self.gradient(X, y, betaOld)
        return betaNew

    def gradient(self, X, y, beta):
        """
        gradient for ols, 
        del(C) = del( (1/n)||X@beta - y||_2^2 = (2/n) X^T @ ( X@beta - y )
        """
        return (2./y.shape[0])* X.T @ (X@beta - y)

    def FindBeta(self,X,y):
        beta = np.zeros(X.shape[1])#.reshape((X.shape[1],1))

        for i in tqdm(range(int(1e9))):
            betaOld = beta
            beta = self.betaNew(X,y,betaOld)
        self.betaBest = beta
        return self.betaBest
