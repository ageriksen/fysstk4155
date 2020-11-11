#!/usr/bin/env python3
import numpy as np

class FFNN:

    def __init__(self, activation, **kwargs):
        self.activation = activation(**kwargs)

    def MakeWeightsAndBiases(self, dim):
        """
        dim denotes the number of nodes per 
        layer, for each index.
        """
        weights = []
        biases = []
        for i in range(len(dim) - 1):
            weights.append(np.random.normal(0,1, (dim[i], dim[i+1])))
            biases.append(np.random.normal(0,1, (dim[i+1], 1)))
        return weights, biases


    def FF(self, x, weights, biases): 
        z_ = [] #lin. combination of previous layer, fed into nodes on next layer
        a_ = [] #activations, values out of each node

        a = x
        a_.append(a) #a_ 1 greater than z_
        for w, b in zip(weights, biases):
            z = w.T @ a + b
            a = self.activation.function(z)

            z_.append(z)
            a_.append(a)
        return a, z_, a_

    def BackPropagate(self, x, y, weights, biases):

        m = y.shape[1] #nr. samples
        num_layers = len(weights) + 1
        grad_weights = []
        grad_biases = []

        y_pred, z_, a_ = self.FF(x, weights, biases)

        #===============================================
        #gradient, weights&biases for last(output) layer
        J = y_pred - y  #derivative of cross-entropy & softmax function
        grad_w = (1./m)*a_[-2]@J.T
        grad_b = (1./m)*np.sum( J, axis=1 ).reshape(-1, 1)

        grad_weights.append(grad_w)
        grad_biases.append(grad_b)
        #===============================================

        for i in reversed(range(num_layers - 2)): #reverse through layers
            J = self.activation.gradient( z_[i] )*( weights[i+1]@J )

            grad_weights.append( (1./m)*(a_[i]@J.T) )
            grad_biases.append( (1./m)*np.sum(J, axis=1).reshape(-1,1) )

        grad_weights.reverse()
        grad_biases.reverse()

        return grad_weights, grad_biases
