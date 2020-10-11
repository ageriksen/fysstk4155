#!/usr/bin/env python3
import numpy as np

class __Regressor:
    
    def __init__(self):
        self

    def fit(self, X_train, y_train):
        raise NotImplementedError("derived class missing fit")

    def predict(self, X_test):
        return X_test @ self.beta

    def get_data(self):
        return self.y_fit, self.beta

    def _svd(self, matrix):
        U, s, VT = np.linalg.svd(matrix)
        D = np.diagflat(s)
        return VT.T @ ( np.linalg.inv(D) @ U.T)

class OLS(__Regressor):

    def __init__(self):
        
        super().__init__()
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train; self.y_train = y_train

        inverse = self._svd(self.X_train.T @ self.X_train)
        self.beta = inverse @ (self.X_train.T @ self.y_train)

        self.y_fit = self.X_train @ self.beta
        
        self.MSE = np.mean( (y_train - self.y_fit)**2 )
        
        
