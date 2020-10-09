from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
np.random.seed(2020)

from sklearn.preprocessing import StandardScaler

def main():
    sys = System()
    sys.generateInputs()

    #solv = Solver(sys, stat)
    solv = Solver(sys)
    solv.setPolyDegree(5)
    solv.solve(solv.noResample)

class System:
    """
    class to hold system info for regression machine learning
    """

    def __init__(self, nrows=100, ncols=100, sigma=1):
        self.nrows = nrows; self.ncols = ncols
        self.sigma = sigma

    def FrankeFunction(self, x , y):
        term1 = 0.75*np.exp( -(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2) )
        term2 = 0.75*np.exp( -((9*x + 1)**2)/49.0 - 0.1*(9*y + 1) )
        term3 = 0.5*np.exp( -(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2) )
        term4 = -0.2*np.exp( -(9*x - 4)**2 - (9*y - 7)**2 )
        return term1 + term2 + term3 + term4
        #self.target = term1 + term2 + term3 + term4

    def generateInputs(self):

        rand_row    =   np.random.uniform(0,1,  size=self.nrows)
        rand_col    =   np.random.uniform(0,1,  size=self.ncols)

        sort_row_index  =   np.argsort(rand_row)
        sort_col_index  =   np.argsort(rand_col)

        rowsort =   rand_row[sort_row_index]
        colsort =   rand_col[sort_col_index]

        self.row_mat, self.col_mat    =   np.meshgrid( colsort, rowsort )
        self.target_mat = self.FrankeFunction(self.row_mat, self.col_mat)
        
        self.row = self.row_mat.ravel()
        self.col = self.col_mat.ravel()
        self.target = self.target_mat.ravel()


class Solver:
    """
    class to solve the regression machine learning problem given the data from 
    the system class
    """

    #def __init__(self, system, stat):
    def __init__(self, system):
        self.System = system #system object passed when initialized. 
        #self.Stat = stat # statisical management object

        self.betas = []
        self.fits = []
        self.preds = []
        self.var_betas = []
        self.test_ratio = 0.2

    def setPolyDegree(self, degree):
        self.polyDegree = degree
        self.Stat = statistics(degree)

    def setSigma(self, sigma):
        self.sigma = sigma

    def setBootstraps(self, bootstraps):
        self.bootstraps = bootstraps

    def setkFolds(self, kfolds):
        self.kfolds = kfolds

    def setTestRatio(self, testRatio):
        self.testRatio = testRatio

    #def resetList(self, lst): #doesn't work in current iteration
    #    lst = []           # seems like the self.object isn't passed, just the values.

    def featureMatrix(self, n):

            N = len(self.System.row)
            l = int((n+1)*(n+2)/2)          # Number of elements in beta                                                               
            X = np.ones((N,l))              # Feature matrix

            for i in range(1,n+1):
                    q = int((i)*(i+1)/2)
                    for k in range(i+1):
                            X[:,q+k] = (self.System.row**(i-k))*(self.System.col**k)
            #self.X = X
            return X

    def split_data(self, data):
        """ 
        takes the data for the problem
        outputs test and training indices for the ratio given.

        The numpy  permutation option randomizes the order of the range given
        and serve as the indices for the slices of our domain we return
        """
        shuffled_indices = np.random.permutation(data.shape[0])
        test_set_size = int(data.shape[0]*self.test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        #return data[train_indices], data[test_indices], target[train_indices], target[test_indices]
        return test_indices, train_indices

    def OLS(self, feature_matrix, targets):
        inverse = np.linalg.pinv(feature_matrix.T @ feature_matrix)
        self.beta = inverse @ (feature_matrix.T @ targets)
        self.var_beta = np.diag(inverse)

    def noResample(self, X, target_train, target_test):
            X_train = X[self.train_indices]
            X_test = X[self.test_indices]
            self.scaler = scale()
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            target_train_scaled = self.scaler.transform(target_train)
            target_test_scaled = self.scaler.transform(target_test)

            self.OLS(X_train_scaled, target_train_scaled)

            target_fit = X_train_scaled @ self.beta
            target_pred = X_test_scaled @ self.beta

            self.betas.append(self.beta)
            self.fits.append(target_fit)
            self.preds.append(target_pred)
            self.var_betas.append( self.System.sigma**2*self.var_beta )

    
    def solve(self, solver):

        self.train_indices, self.test_indices = self.split_data(self.System.target)

        target_train = self.System.target[self.train_indices]
        target_test = self.System.target[self.test_indices]

        for deg in range(self.polyDegree):
            #X = create_X(rowdata, coldata, deg)
            X = self.featureMatrix(deg)
            solver(X, target_train, target_test)
            
            self.Stat.store(deg, target_train, target_test, self.fits, self.preds)

        self.Stat.plot2D()

class statistics:

    def __init__(self, poly):

        self.complexity = np.arange(poly)
        self.MSEfit = np.zeros(poly)
        self.MSEpred =  np.zeros(poly)
        self.R2fit =  np.zeros(poly)
        self.R2pred = np.zeros(poly)

    def store(self, deg, train, test, fit, pred):
        self.MSEfit[deg] = self.MSE(train, fit)
        self.MSEpred[deg] = self.MSE(test, pred)
        self.R2fit[deg] = self.R2(train, fit)
        self.R2pred[deg] = self.R2(test, pred)


    def R2(self, target, model):
        return 1 - ( np.sum( (target-model)**2 )/np.sum( (target-np.mean(target))**2 ) )

    def MSE(self, target, model): 
        #n = np.size(target)
        #return np.sum( (target-model)**2 )/n
        #return np.mean( (target - model)**2 ) 
        return np.mean( np.mean(    (target - model)**2, axis=1, keepdims=True ) )

    def BIAS2(self, data, model):
        return np.mean( (data - np.mean(model, axis=1, keepdims=True))**2   )

    def VARIANCE(self, model):
        return np.mean( np.var( model, axis=1, keepdims=True  )   )

    #def plot2D(self, x, ylist, ylegends, xlabel, ylabel, title=False):
    def plot2D(self, title=False):
        plt.figure()
        #for i in range(len(ylist)):
            #plt.plot(x, ylist[i], label=ylegends[i])
        plt.plot(self.complexity, self.MSEfit, label='MSE train')
        plt.plot(self.complexity, self.MSEpred, label='MSE test')
        plt.legend()
        plt.xlabel('model complexity')
        plt.ylabel('MSE')
        if title!=bool:
            plt.title(title)
        plt.show()

    def plot3D(self, x, y, z, zlim_min=-.10, zlim_max=1.40 ):
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
    

class scale:

    def __init__(self):
        self

    def fit(self, fitting):
        self.mean = np.mean(fitting)

    def transform(self, data):
        return data - self.mean


#####these are probably best left in a "statistics" class of some sort#####
#def noResampling(rowdata, coldata, target, maxdegree, sigma=1):
#    MSEfit = np.zeros(maxdegree)
#    MSEpred =  np.zeros(maxdegree)
#    R2fit =  np.zeros(maxdegree)
#    R2pred = np.zeros(maxdegree)
#
#
#    plotlist = [MSEfit, MSEpred]
#    legendlist = ['train', 'test']
#    plot2D(np.arange(maxdegree), plotlist, legendlist, 'model complexity', 'MSE', 'Franke function no resampling')

if __name__=='__main__':
    main()
