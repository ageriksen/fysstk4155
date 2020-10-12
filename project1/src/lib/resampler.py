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

    def __init__(self, regressor, designmatrix, row, col, targets):
        self.regressor = regressor
        self.designmatrix = designmatrix
        self.row = row
        self.col = col
        self.targets = targets



    def run(self, polydegree, test_ratio=0.2):

        X = self.designmatrix(self.row, self.col, polydegree)

        traintest = TrainTest()
        traintest.indices(self.targets)
        X_train, X_test, y_train, y_test = traintest.split(X, self.targets)

        self.regressor.fit(X_train, y_train)
        self.regressor.predict(X_test, y_test)

