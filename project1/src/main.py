import numpy as np
import matplotlib.pyplot as plt

from lib.resampler import NoResample
from lib.regressor import OLS
from lib.traintest import TrainTest
from lib.franke import FrankeFunction
from lib.designmatrix import DesignMatrix


def solve_franke(regressor, resampler, **kwargs):
    rows = 100; cols=200; sigma = .1; poly_max = 10

    rand_row        =   np.random.uniform(0,1,  size=rows)
    rand_col        =   np.random.uniform(0,1,  size=cols)

    sort_row_index  =   np.argsort(rand_row); 
    sort_col_index  =   np.argsort(rand_col)

    rowsort         =   rand_row[sort_row_index]; 
    colsort         =   rand_col[sort_col_index]

    row_mat, col_mat=   np.meshgrid( colsort, rowsort )

    franke          =   FrankeFunction(row_mat, col_mat) + sigma*np.random.randn(rows,cols)

    #data            =   np.asarray((row_mat.ravel(), col_mat.ravel()))
    row             =   row_mat.ravel()
    col             =   col_mat.ravel()
    target          =   franke.ravel()


    results = []
    
    if 'regressor' in kwargs: regr = regressor(kwargs['regressor'])
    else: regr = regressor()
    if 'resampler' in kwargs: 
        resample = resampler(regr, DesignMatrix, row, col, target, kwargs['resampler'])
    else: resample = resampler(regr, DesignMatrix, row, col, target)

    for deg in range(poly_max):
        resample.run(poly_max)
        results.append(regr.get_data())

    if any( 'MSEtrain' in stat for stat in results ):
        polydegrees = np.arange(poly_max)
        plt.figure()
        plt.plot(polydegrees, np.asarray([stat['MSEtrain'] for stat in results]), label='train')
        plt.plot(polydegrees, np.asarray([stat['MSEtest'] for stat in results]), label='test')
        plt.legend()
        plt.title('No resampling, OLS, noise={:.2f}'.format(sigma))
        plt.xlabel('model polynomial degree')
        plt.ylabel('MSE')
        plt.show()

if __name__ == '__main__':
    solve_franke(OLS, NoResample)
