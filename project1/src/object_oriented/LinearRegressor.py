
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
np.random.seed(2020)

from sklearn.preprocessing import StandardScaler


def FrankeFunction(x,y):
    term1 = 0.75*np.exp( -(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2) )
    term2 = 0.75*np.exp( -((9*x + 1)**2)/49.0 - 0.1*(9*y + 1) )
    term3 = 0.5*np.exp( -(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2) )
    term4 = -0.2*np.exp( -(9*x - 4)**2 - (9*y - 7)**2 )
    return term1 + term2 + term3 + term4


class LinReg:

    def __init__(self):
        self

    def OLS(self, rows, cols, poly, target):
        msetrain = np.zeros(poly)
        msetest = np.zeros(poly)

        train_indices, test_indices = self.trainTest(target)
        target_train = target[train_indices]
        target_test = target[test_indices]

        for deg in range(poly):
            X = self.featureMatrix(rows, cols, deg)

            X_train=X[train_indices]; X_test=X[test_indices]

            mean = np.mean(X_train)
            X_train_scaled = X_train - mean
            X_test_scaled = X_test - mean
            target_train_scaled = target_train - mean
            target_test_scaled = target_test - mean
            X_test_scaled[0,:] = 1
            X_train_scaled[0,:] = 1

            inverse = np.linalg.pinv(X_train_scaled.T @ X_train_scaled)
            beta = inverse @(X_train_scaled.T @ target_train_scaled)
    
            fit = X_train_scaled @ beta
            pred= X_test_scaled @ beta




            msetrain[deg] = np.mean( (target_train_scaled - fit)**2 )
            msetest[deg] = np.mean( (target_test_scaled - pred)**2 )


    def featureMatrix(self, x, y, n ):
            if len(x.shape) > 1:
                    x = np.ravel(x)
                    y = np.ravel(y)

            N = len(x)
            l = int((n+1)*(n+2)/2)          # Number of elements in beta                                                               
            X = np.ones((N,l))

            for i in range(1,n+1):
                    q = int((i)*(i+1)/2)
                    for k in range(i+1):
                            X[:,q+k] = (x**(i-k))*(y**k)
            return X

    def trainTest(self, data, test_ratio=0.2):
        """ 
        takes the data for the problem
        outputs test and training indices for the ratio given.

        The numpy  permutation option randomizes the order of the range given
        and serve as the indices for the slices of our domain we return
        """
        shuffled_indices = np.random.permutation(data.shape[0])
        test_set_size = int(data.shape[0]*test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return test_indices, train_indices

if __name__ == '__main__':

    polydegree = 10; sigma = 1
    nrows = 100; ncols = 200

    rand_row    =   np.random.uniform(0,1,  size=nrows)
    rand_col    =   np.random.uniform(0,1,  size=ncols)

    sort_row_index  =   np.argsort(rand_row)
    sort_col_index  =   np.argsort(rand_col)

    rowsort =   rand_row[sort_row_index]
    colsort =   rand_col[sort_col_index]

    row_mat, col_mat    =   np.meshgrid( colsort, rowsort )

    franke = FrankeFunction(row_mat, col_mat) \
            +   sigma*np.random.randn(nrows,ncols)

    row_arr = row_mat.ravel()
    col_arr = col_mat.ravel()
    target = franke.ravel()

    regression = LinReg()
    regression.OLS(row_arr, col_arr, polydegree, target)
