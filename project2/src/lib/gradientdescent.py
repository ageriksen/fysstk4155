#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm

class _gradientDescent:
    #TODO Gradient is too specific. need to implement setting
    #       it to given class object. 
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
    def __init__(self, *args):
        super().__init__(*args)
        self.learning = None
        self.t = None
        self.lrnmode = {\
                'static': self.static, 
                'dynamic': self.dynamic}

    def SetLearningMode(self, mode, t):
        """
        sets the learning rate to either 'static' or 'dynamic' and the 
        rate to either the static fraction t, or the list of the t0 and t1 
        for the dynamic case
        """
        assert mode in self.lrnmode, mode + " input, needs 'static' or 'dynamic'"
        self.learning = self.lrnmode[mode]
        self.t = t

    def dynamic(self, t): return self.t[0] / (t + self.t[1])
    def static(self, t): return self.t

    def fit(self, X, y, minibatches, epochs, lrn):
        assert self.learning is not None, "need to set learning mode"

        self.null = False
        batchsize = int(y.shape[0]/minibatches)
        theta = np.random.randn(X.shape[1])
        self.t = lrn
        for epoch in range(epochs):
            for batch in range(minibatches):
                rand = np.random.randint(minibatches)
                batch = rand*batchsize
                Xi = X[batch:batch+batchsize]; yi = y[batch:batch+batchsize]
                gradients = self.gradient(Xi, yi, theta) 
                eta = self.learning(epoch*minibatches+batch) 
                theta -= eta*gradients
            if self.null:
                print("found a 0")
                break
        self.theta = theta

class generalSGD:

    def fit(self, X, y, batches, epochs, learningrate):

        batchsize = int(float(y.shape[0])/minibatches)#force float division, then take lowest integer
        for epoch in range(epochs):
            for batch in range(batches):
                rand = np.random.randint(minibatches)
                minibatch = rand*batchsize
                Xi = X[minibatch:minibatch+batchsize]
                yi = y[minibatch:minibatch+batchsize]
                gradients = self.gradient(Xi, yi, eta)
                theta -= eta*gradients
        
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
