#!/usr/bin/env python3

import numpy as np

class _activationBase:

    def __init__(self):
        self

    def function(self):
        raise NotImplementedError

    def gradient(self):
        raise NotImplementedError

class Sigmoid(_activationBase):
    
    def __init__(self):
        self


    def function(self, x):
        return 1./( 1 + np.exp(-x) )

    def gradient(self, x): 
        return self.function(x)*( 1 - self.function(x) )
