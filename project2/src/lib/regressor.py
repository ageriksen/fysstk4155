#!/usr/bin/env python3
import numpy as np

class _baseregressor:

    def __init__(self):
        self

    def fit(self):
        raise NotImplementedError


class OLS(_baseregressor):

    def __init__(self):
        self

    def fit(self, X, y):
        covar = svdinv( X.T @ X )
        self.beta = covar @ (X.T @ y)
        return self.beta

    def predict(self, X):
        try:
            return X @ self.beta
        except NameError:
            print("no model fit")



def svdinv(M):
    U, s, VT = np.linalg.svd(M)
    D = np.diagflat(s)
    return VT.T @ ( np.linalg.pinv(D) @ U.T)
