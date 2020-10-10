import numpy as np
import matplotlib.pyplot as plt


def FrankeFunction(x,y):
    term1 = 0.75*np.exp( -(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2) )
    term2 = 0.75*np.exp( -((9*x + 1)**2)/49.0 - 0.1*(9*y + 1) )
    term3 = 0.5*np.exp( -(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2) )
    term4 = -0.2*np.exp( -(9*x - 4)**2 - (9*y - 7)**2 )
    return term1 + term2 + term3 + term4

def trainTest(data, test_ratio=0.2):
    
    shuffled_indices = np.random.permutation(data.shape[0])
    test_set_size = int(data.shape[0]*test_ratio)
    
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return test_indices, train_indices


def featureMatrix(x, y, n ):
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

def model(rows, cols, target, polydegree, resampler, resampling, regressor):

    train, test = trainTest(target)
    resultlist = np.zeros(polydegree)
    for deg in range(polydegree):
        X = featureMatrix(rows, cols, deg)
        resultlist[deg] = resampler(regressor, resampling, train, X, target)
    #plot(np.arange(polydegree), resultlist)

def bootstrap(regressor, bootstraps, train_indices, X, y):
    #resampler
    bias = []; variance=[]; error=[]
    fits = []; preds = []; betas = []
    for boot in range(bootstraps):
        train, test = trainTest(train_indices)
        X_train = X[train]; X_test = X[test]
        y_train = y[train]; y_test = y[test]


        beta, var_beta = regressor(X_train, y_train)

        fit = X_train @ beta
        pred = X_test @ beta

        preds.append(pred)

    bias.append(np.mean( (target[test] - np.mean(preds, axis=1, keepdims=True))**2   ))
    variance.append(np.mean( np.var( preds, axis=1, keepdims=True  )   ))
    error.append(bias[-1] + variance[-1])

    return (bias, 'bias'), (variance, 'variance'), (error, 'error')

def OLS(X, y):
    #regressor
    inv = np.linalg.inv(X.T@X)
    beta = inv @ (X.T @ y)
    beta_var = np.diag(inv)
    return beta, beta_var

def plot(x, variables):
    plt.figure()
    for i in range(len(variables)):
        plt.plot(x, variables[i][0], label=variables[i][1])
    plt.legend()
    plt.show()

if __name__ == '__main__':

    polydegree = 10; bootstraps = 100; sigma = .1
    nrows = 100; ncols = 100
    kfolds = 10

    rand_row    =   np.random.uniform(0,1,  size=nrows)
    rand_col    =   np.random.uniform(0,1,  size=ncols)

    sort_row_index  =   np.argsort(rand_row)
    sort_col_index  =   np.argsort(rand_col)

    rowsort =   rand_row[sort_row_index]
    colsort =   rand_col[sort_col_index]

    row_mat, col_mat    =   np.meshgrid( colsort, rowsort )

    franke = FrankeFunction(row_mat, col_mat) \
            +   sigma*np.random.randn(nrows,ncols)

    rows = row_mat.ravel()
    cols = col_mat.ravel()
    target = franke.ravel()

    model(rows, cols, target, polydegree, bootstrap, bootstraps, OLS)
