from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
np.random.seed(2020)

from sklearn.preprocessing import StandardScaler

def main():
    polydegree = 10; bootstraps = 100; sigma = .1
    nrows = 100; ncols = 100
    kfolds = 10

    row, col, franke = makeFranke()
    #row, col, franke = makeFranke(nrows, ncols, sigma)

    #noResampling(row, col, franke, polydegree)

    #Bootstrap(row, col, franke, polydegree, bootstraps, sigma)
    #Bootstrap(row, col, franke, polydegree, bootstraps)

    kfoldCV(row, col, franke, polydegree, kfolds)

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
    #return np.mean( (target - model)**2 ) 
    return np.mean( np.mean(    (target - model)**2, axis=1, keepdims=True ) )

def BIAS2(data, model):
    return np.mean( (data - np.mean(model, axis=1, keepdims=True))**2   )

def VARIANCE(model):
    return np.mean( np.var( model, axis=1, keepdims=True  )   )

def plot2D(x, ylist, ylegends, xlabel, ylabel, title=False):
    plt.figure()
    for i in range(len(ylist)):
        plt.plot(x, ylist[i], label=ylegends[i])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title!=bool:
        plt.title(title)
    plt.show()

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
    #plot3D(row_mat, col_mat, franke)
    return row_mat.ravel(), col_mat.ravel(), franke.ravel()

def OLS(feature_matrix, targets):
    inverse = np.linalg.pinv(feature_matrix.T @ feature_matrix)
    beta = inverse @ (feature_matrix.T @ targets)
    #return beta, np.diag(inverse)
    return beta

def noResampling(rowdata, coldata, target, maxdegree, sigma=1):

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
        X_train[0,:] = 1
        X_test[0,:] = 1
        #print(X_train)

        inverse = np.linalg.pinv(X_train.T @ X_train)
        #beta = inverse @ (X_train.T @ target_train )
        beta = OLS(X_train, target_train)
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
    
    plotlist = [MSEfit, MSEpred]
    legendlist = ['train', 'test']
    plot2D(np.arange(maxdegree), plotlist, legendlist, 'model complexity', 'MSE', 'Franke function no resampling')

def Bootstrap(rowdata, coldata, target, maxdegree, bootstraps, sigma=1):

    #mse = np.zeros((bootstraps, maxdegree))
    #err = np.zeros((bootstraps, maxdegree)) 
    #bias= np.zeros((bootstraps, maxdegree))
    #var = np.zeros((bootstraps, maxdegree))
    err_train = np.zeros(maxdegree) 
    bias_train= np.zeros(maxdegree)
    var_train = np.zeros(maxdegree)
    err_test = np.zeros(maxdegree) 
    bias_test= np.zeros(maxdegree)
    var_test = np.zeros(maxdegree)


    train_indices, test_indices = split_data(target)
    target_TRAIN = scale(target[train_indices])
    target_TEST = scale(target[test_indices])

    for deg in range(maxdegree):
        X = create_X(rowdata, coldata, deg)
        fits = []; preds = []; betas = []

        for boot in range(bootstraps):
            train_indices, test_indices = split_data(target_TRAIN)

            target_train = scale(target_TRAIN[train_indices])
            target_test = scale(target_TRAIN[test_indices])

            X_train = scale(X[train_indices])
            X_test = scale(X[test_indices])
            X_train[0,:] = 1
            X_test[0,:] = 1

            inverse = np.linalg.pinv(X_train.T @ X_train)
            #beta = inverse @ (X_train.T @ target_train )
            beta = OLS(X_train, target_train)
            target_fit = X_train @ beta
            target_pred = X_test @ beta

            fits.append(target_fit)
            preds.append(target_pred)
            betas.append(beta)


        bias_train[deg]= BIAS2(target_train, fits)
        var_train[deg] = VARIANCE(fits)
        err_train[deg] = bias_train[deg] + var_train[deg]

        bias_test[deg]= BIAS2(target_test, preds)
        var_test[deg] = VARIANCE(preds)
        err_test[deg] = bias_test[deg] + var_test[deg]

    plotlist = [err_train, bias_train, var_train]
    legendlist = ['error', 'bias^2', 'variance']
    plot2D(np.arange(maxdegree), plotlist, legendlist, 'model complexity', '', 'Bootstrap Bias-Variance Franke Train')
    plotlist = [err_test, bias_test, var_test]
    legendlist = ['error', 'bias^2', 'variance']
    plot2D(np.arange(maxdegree), plotlist, legendlist, 'model complexity', '', 'Bootstrap Bias-Variance Franke Test')
    
def kfoldCV(rowdata, coldata, target, maxdegree, folds, sigma=1):
    
    mse_train = []
    mse_tests = []
    
    train_indices, test_indices = split_data(target)
    target_train = scale(target[train_indices])
    target_test = scale(target[test_indices])

    foldsize = int(len(target_train)/folds)

    #write in foldsize, pick out k-fold and the rest from train_indices => np.delete(target, k-th fold) will do the trick
    #should the scaling be per k-fold training set, or should it be done early? 

    for deg in range(maxdegree):
        X = create_X(rowdata, coldata, deg)
        fits = []; preds = []; betas = []; k_tests = []; k_train = []
    
        

        for k in range(folds):

            kstart = k*foldsize
            kstop = kstart+foldsize
            foldindices = np.arange(kstart, kstop)

            k_fold_indices = train_indices[foldindices]
            k_train_indices = np.delete(train_indices, foldindices)
            
            k_target_train = target[k_train_indices]
            k_target_test = target[k_fold_indices]

            k_X_train = scale(X[k_train_indices])
            k_X_test = scale(X[k_fold_indices])
            k_X_train[0,:] = 1
            k_X_test[0,:] = 1

            inverse = np.linalg.pinv(k_X_train.T @ k_X_train)
            #beta = inverse @ (k_X_train.T @ k_target_train )
            beta = OLS(k_X_train, k_target_train)
            target_fit = k_X_train @ beta
            target_pred = k_X_test @ beta

            fits.append(target_fit)
            preds.append(target_pred)
            betas.append(beta)
            k_tests.append(k_target_test)
            k_train.append(k_target_train)



        mse_train.append(MSE(np.array(k_train), np.array(fits)))
        mse_tests.append(MSE(np.array(k_tests), np.array(preds)))

    plot2D(np.arange(maxdegree), [mse_train, mse_tests], ['train', 'test'], 'model complexity', 'MSE')

if __name__ == '__main__':
    main()
