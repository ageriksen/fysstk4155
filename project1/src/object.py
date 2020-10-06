from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from random import random, seed
np.random.seed(2020)

def main():
    row, col, franke = makeFranke()
    return


def FrankeFunction(x,y):
    term1 = 0.75*np.exp( -(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2) )
    term2 = 0.75*np.exp( -((9*x + 1)**2)/49.0 - 0.1*(9*y + 1) )
    term3 = 0.5*np.exp( -(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2) )
    term4 = -0.2*np.exp( -(9*x - 4)**2 - (9*y - 7)**2 )
    return term1 + term2 + term3 + term4

def create_X(x, y, n ):
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

#def split_data(data, target, test_ratio=0.2):
def split_data(data, test_ratio=0.2):
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
    #return data[train_indices], data[test_indices], target[train_indices], target[test_indices]
    return test_indices, train_indices

def scale(data): 
    return data - np.mean(data) 

def R2(target, model):
    return 1 - ( np.sum( (target-model)**2 )/np.sum( (target-np.mean(target))**2 ) )

def MSE(target, model): 
    #n = np.size(target)
    #return np.sum( (target-model)**2 )/n
    return np.mean( (target - model)**2 ) 

########## These are likely problematic irt. overhead when running. 
def ERROR(data, model): 
    return np.mean( np.mean(    (data - model)**2, axis=1, keepdims=True    )   )

def BIAS(data, model):
    return np.mean( (data - np.mean(model, axis=1, keepdims=True))**2   )

def VARIANCE(model):
    return np.mean( np.var( model, axis=1, keepdims=True    )   )
#########

def plot3D(x, y, z, zlim_min=-.10, zlim_max=1.40 ):
    #Create figures
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(zlim_min, zlim_max)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def makeFranke(rows=100, cols=200, sigma=1):
    rand_row    =   np.random.uniform(0,1,  size=rows)
    rand_col    =   np.random.uniform(0,1,  size=cols)

    sort_row_index  =   np.argsort(rand_row)
    sort_col_index  =   np.argsort(rand_col)

    rowsort =   rand_row[sort_row_index]
    colsort =   rand_col[sort_col_index]

    row_mat, col_mat    =   np.meshgrid( colsort, rowsort )

    franke = FrankeFunction(row_mat, col_mat) \
            +   sigma*np.random.randn(rows,cols)
    plot3D(row_mat, col_mat, franke)
    return row_mat.ravel(), col_mat.ravel(), franke.ravel()

def OLS(rowdata, coldata, target, maxdegree, sigma):

    betas = []
    fits = []
    preds = []
    var_betas = []

    MSEfit = np.zeros(maxdegree)
    MSEpred =  np.zeros(maxdegree)
    R2fit =  np.zeros(maxdegree)
    R2pred = np.zeros(maxdegree)

    train_indices, test_indices = split_data(target)
    target_train = scale(target[train_indices])
    target_test = scale(target[test_indices])
    for deg in range(maxdegree):
        X = create_X(rowdata, coldata, deg)
        X_train = scale(X[train_indices])
        X_test = scale(X[test_indices])

        inverse = np.linalg.pinv(X_train.T @ X_train)
        beta = inverse @ (X_train.T @ target )
        target_fit = X_train @ beta
        target_pred = X_test @ beta

        betas.append(beta)
        fits.append(target_fit)
        preds.append(target_pred)
        var_betas.append( sigma**2*np.diag(inverse) )
        MSEfit[deg] = MSE(target_train, target_fit)
        MSEpred[deg] = MSE(target_test, target_pred)
        R2fit[deg] = R2(target_train, target_fit)
        R2pred[deg] = R2(target_test, target_pred)



if __name__ == '__main__':
    main()
