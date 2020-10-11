from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
np.random.seed(2020)

from sklearn.preprocessing import StandardScaler

from lib.regressor import OLS
from lib.designmatrix import DesignMatrix
from lib.franke import FrankeFunction
from lib.traintest import TrainTest

class NoResample:

    def __init__(self, regressor, designmatrix, inputs, targets):
        self.regressor = regressor
        self.designmatrix = designmatrix
        self.inputs = inputs
        self.targets = targets



    def validate(self, polydegree, test_ratio=0.2):

        X = self.designmatrix(self.inputs[0], self.inputs[1], polydegree)

        traintest = TrainTest()
        traintest.indices(self.targets)
        X_train, X_test, y_train, y_test = traintest.split(X, self.targets)

        self.regressor.fit(X_train, y_train)

