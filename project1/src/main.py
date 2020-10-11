import numpy as np

from lib.resampler import NoResample
from lib.regressor import OLS
from lib.traintest import TrainTest
from lib.franke import FrankeFunction
from lib.designmatrix import DesignMatrix


rows = 100; cols=100; sigma = .1
poly_max = 5

rand_row    =   np.random.uniform(0,1,  size=rows)
rand_col    =   np.random.uniform(0,1,  size=cols)

sort_row_index  =   np.argsort(rand_row); sort_col_index  =   np.argsort(rand_col)

rowsort =   rand_row[sort_row_index]; colsort =   rand_col[sort_col_index]

row_mat, col_mat    =   np.meshgrid( colsort, rowsort )

franke = FrankeFunction(row_mat, col_mat) + sigma*np.random.randn(rows,cols)

data = np.asarray((row_mat.ravel(), col_mat.ravel()))
target = franke.ravel()

reg = NoResample(OLS(), DesignMatrix, data, target)
reg.validate(poly_max)
