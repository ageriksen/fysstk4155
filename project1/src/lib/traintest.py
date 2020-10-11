#!/usr/bin/env python3

import numpy as np

class TrainTest:
    
    def __init__(self, ratio=0.2):
        self.test_ratio = ratio

    def indices(self, target):
        shuffled_indices = np.random.permutation(target.shape[0])
        test_set_size = int(target.shape[0]*self.test_ratio)
        #return np.split(shuffled_indices, [test_set_size])
        self.train, self.test = np.split(shuffled_indices, [test_set_size])

    def split(self, X, y):
        return X[self.train], X[self.test], y[self.train], y[self.test]
