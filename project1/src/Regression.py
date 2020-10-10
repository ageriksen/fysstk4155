from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
np.random.seed(2020)

from sklearn.preprocessing import StandardScaler

class Regression:

    def __init__(self, system):
        self.test_ratio=0.2
        self.system = system

    def trainTestSplit(self, target):
        shuffled_indices = np.random.permutation(target.shape[0])
        test_set_size = int(target.shape[0]*self.test_ratio)
        return np.split(shuffled_indices, [test_set_size])


    def noResample(self, featureMatrix, target):
        train, test = self.trainTestSplit(target)
        feature_train = featurematrix[train]
        feature_test = featurematrix[test]
        target_train = target[train]
        target_test = target[test]

        beta = np.linalg.pinv( feature_train.T@feature_train )\
                @( feature_train.T@target_train )
        fit = feature_train @ beta
        pred = feature_test @ beta

        self.system.fit = fit
        self.system.pred = pred



class System:

    def __init__(self):
        self.fit = np.empty(0)
        self.pred = np.empty(0)
        self.beta = np.empty(0)
        self.train = np.empty(0)


if __name__ == '__main__':
    reg = Regression()
    sys = System()
