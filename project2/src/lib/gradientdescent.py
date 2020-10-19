#!/usr/bin/env python3

class SGD:

    def __init__(self):
        self

    def betaNew(self, X, y, betaOld):
        """

        iteration of the gradient for beta's pure pseudocode as of now
        takes the current/old beta and returns the new beta.
        This is currently a vector operation, so this would compute the
        entire beta
        """
        #betaNew = betaOld - inverse(X.T@W@X) @ (-X.T@(y-p)) # computed w/ old beta
        
        betaNew = betaOld - learningrate*gradient(X, y, betaOld)

    def gradient(self, X, y, beta):
        """
        gradient for ols, 
        del(C) = del( (1/n)||X@beta - y||_2^2 = (2/n) X^T @ ( X@beta - y )
        """
        return (2/n)* X.T @ (X@beta - y)
