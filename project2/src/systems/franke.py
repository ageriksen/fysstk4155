#!/usr/bin/env python3
import lib.regressor as reg
import lib.gradientdescent as gd

import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class _BaseFrankeRegressor:

    def __init__(self):
        self.setup = False

    def PolyFeatures(self, x, y, n ):
        """
        creates a feature matrix for polynomial regression
        of 2 dependent variables to n'th degree. 
        """
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


    def R2(self, target, model):
        return 1 - ( np.sum( (target-model)**2 )/np.sum( (target-np.mean(target))**2 ) )

    def singleMSE(self, target, model): 
        return np.mean( (target - model)**2 ) 

    def multiMSE(self, target, model):
        assert model.shape[1] > 1, "use the single version instead"
        return np.mean( np.mean(    (target - model)**2, axis=1, keepdims=True ) )

    def TrainTestSplit(self, data, test_ratio=.2):
        """
        takes data array and returns randomised
        set of indices for train and test splits
        """
        shuffle = np.random.permutation(data.shape[0])
        test_size = int(data.shape[0]*test_ratio)
        self.test = shuffle[:test_size] # [test-size|               ]
        self.train = shuffle[test_size:]# [         |   train size  ]
        return self.train, self.test

    def FrankeFunction(self, x,y):
        term1 = 0.75*np.exp(               
                - (0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2) )
        term2 = 0.75*np.exp(            
                - ((9*x + 1)**2)/49.0 - 0.1*(9*y + 1) )
        term3 = 0.5*np.exp(              
                - (9*x - 7)**2/4.0    - 0.25*((9*y - 3)**2) )
        term4 = -0.2*np.exp(              
                - (9*x - 4)**2        - (9*y - 7)**2 )

        return term1 + term2 + term3 + term4

    def SetSystem(self, rows, cols, sigma): 
        row = np.linspace(0,1,rows)
        col = np.linspace(0,1,cols)
        self.row_mat, self.col_mat = np.meshgrid(col, row)
        self.height_mat = \
                self.FrankeFunction(self.row_mat, self.col_mat)\
                + sigma*np.random.randn(rows, cols)
        self.row = self.row_mat.ravel()
        self.col = self.col_mat.ravel()
        self.height = self.height_mat.ravel()
        self.setup = True



class FrankeRegression(_BaseFrankeRegressor):
    """
    class for performing Linear regression on the Franke function.
    I think the specific regressor(OLS, SGD, etc.) should be made 
    outside the class, as this is mainly to set up and keep the necessary 
    values, x,y,z coordinates and such. 
    """

    def __init__(self):
        self.regressor = None
        self.sklregressor = None
        self.resamplers = {'None': self.NoResample}
        self.resampler = None

    def SetResampler(self, resampler, **kwargs):
        assert resampler in self.resamplers, resampler + " not implemented"
        self.resampler = self.resamplers[resampler]

    def SetRegressor(self, regressor, **kwargs):
        self.regressor = regressor(**kwargs)

    def SetSklRegressor(self, sklregressor, **kwargs):
        self.sklregressor = sklregressor(**kwargs)

    def Run(self,  testRatio=.2, maxdegree=10):
        assert self.setup, "You need to setup the system!"
        assert self.regressor is not None, "Regressor not set"
        assert self.sklregressor is not None, "skl regressor not set"
        assert self.resampler is not None, "resampling method not set"

        self.regtrain        =   []
        self.regtest         =   []
        self.regR2           =   []
        self.regTrainMSE     =   []
        self.regTestMSE      =   []

        self.skltrain        =   []
        self.skltest         =   []
        self.sklR2           =   []
        self.sklTrainMSE     =   []
        self.sklTestMSE      =   []

        for deg in tqdm(range(1, maxdegree)):

            features = self.PolyFeatures(self.row, self.col, deg)

            #TODO   The feature matrix is the only thing needing to be
            #       set in this instance. eventual hyperparameters and
            #       resampling could be set before and after this section

            self.resampler(features, testRatio)


        self.regtrain        =   np.array(   regtrain        )
        self.regtest         =   np.array(   regtest         )
        self.regR2           =   np.array(   regR2           )
        self.regTrainMSE     =   np.array(   regTrainMSE     )
        self.regTestMSE      =   np.array(   regTestMSE      )
        self.skltrain        =   np.array(   skltrain        )
        self.skltest         =   np.array(   skltest         )
        self.sklR2           =   np.array(   sklR2           )
        self.sklTrainMSE     =   np.array(   sklTrainMSE     )
        self.sklTestMSE      =   np.array(   sklTestMSE      )


        degrees = np.arange(1, maxdegree)
        plt.plot(degrees, self.sklTrainMSE, label='skl train')
        plt.plot(degrees, self.sklTestMSE, label='skl test')
        plt.plot(degrees, self.regTrainMSE, label='SGD train')
        plt.plot(degrees, self.regTestMSE, label='SGD test')
        plt.legend()
        plt.xlabel('model complexity')
        plt.ylabel('MSE')
        plt.show()


    def NoResample(self, features, testRatio):

            train, test = self.TrainTestSplit(self.height,testRatio)

            heightTrain     =   self.height[train]
            heightTest      =   self.height[test]

            featuresTrain   =   features[train]
            featuresTest    =   features[test]

            #TODO normalize features and targets

            #
            self.sklregressor.fit(featuresTrain, heightTrain)

            self.skltrain.append(self.sklregressor.predict(featuresTrain))
            self.skltest.append(self.sklregressor.predict(featuresTest))
            
            self.sklR2.append(self.sklregressor.score(featuresTest, heightTest))
            self.sklTrainMSE.append(mean_squared_error(heightTrain, self.skltrain[-1]))
            self.sklTestMSE.append(mean_squared_error(heightTest, self.skltest[-1]))

            #
            self.regressor.fit(featuresTrain, heightTrain)

            self.regtrain.append(featuresTrain@self.regressor.theta)
            self.regtest.append(featuresTest@self.regressor.theta)

            self.regR2.append(self.R2(heightTest, self.regtest[-1]))
            self.regTrainMSE.append(self.singleMSE(heightTrain, self.regtrain[-1]))
            self.regTestMSE.append(self.singleMSE(heightTest, self.regtest[-1]))
