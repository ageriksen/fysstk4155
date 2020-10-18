#!/usr/bin/env python3

class SGD:

    def __init__(self):
        self

    def betaNew(self):
        """
        Using the Newton-Rapson method for the gradient of the 
        cost function
        use taylor
        f(s) = 0 = f(x) + (s-x)f'(x) + (s-x)^2/2 f''(x)...
        for well behaved funcs,
        f(x) + (s-x)f'(x) ~= 0
        s ~= x - f(x)/f'(x)

        x[i+1] = x[i] + f(x[i])/f'(x[i])

        x' = ( x(t+h) - x(t) ) / h = v
        x[i+1] = x[i] + v[i]*h
        v[i+1] = (x[i+1] - x[i])/h


        iteration of the gradient for beta's pure pseudocode as of now
        takes the current/old beta and returns the new beta.
        This is currently a vector operation, so this would compute the
        entire beta
        """
        betaNew = betaOld - inverse(X.T@W@X) @ (-X.T@(y-p)) # computed w/ old beta
        
