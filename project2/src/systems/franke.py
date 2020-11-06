#!/usr/bin/env python3
import lib.regressor as reg
import lib.gradientdescent as gd

import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#def R2(self, target, model):
#    return 1 - ( np.sum( (target-model)**2 )/np.sum( (target-np.mean(target))**2 ) )
#
#def singleMSE(self, target, model): 
#    return np.mean( (target - model)**2 ) 
#
#def multiMSE(self, target, model):
#    assert model.shape[1] > 1, "use the single version instead"
#    return np.mean( np.mean(    (target - model)**2, axis=1, keepdims=True ) )


class FrankeRegression:
    """
    class for performing Linear regression on the Franke function.
    I think the specific regressor(OLS, SGD, etc.) should be made 
    outside the class, as this is mainly to set up and keep the necessary 
    values, x,y,z coordinates and such. 
    """

    def __init__(self):
        self.regressor = None
        self.sklregressor = None
        self.hyperParameters = [None]
        self.resampler = None
        self.testRatio = .2

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

    def SetHyperParameter(self, hyperParameters):
        self.hyperParameters = hyperParameters

    def SetRegressor(self, regressor, **kwargs):
        self.regressor = regressor(**kwargs)

    def SetLearningMode(self, mode, t):
        self.regressor.SetLearningMode(mode, t)

    def SetSklRegressor(self, sklregressor, **kwargs):
        self.sklregressor = sklregressor(**kwargs)

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

    def Run(self,  testRatio=.2, maxdegree=10):
        """
        Running the regression of the franke system set up previosly.
        The methods for resampling differs a bit, so the regressor and
        the resampler are separate functions (the regressor being it's 
        own class). 

        The values that come into the regressor, such as 
        the hyper parameters and the set of features and outputs selected
        for the current round of resampling are found from the main class. 
        the hyperparameter is by default a list of a single nonetype object,
        which can be set do otherwise before running the regression. 
        
        The values for the resampler is also set beforehand, likely to be
        put in a dict. 
        """
        #assert self.resampler is not None, "resampling method not set"
        assert self.regressor is not None, "Regressor not set"
        assert self.sklregressor is not None, "skl regressor not set"
        assert self.setup, "You need to setup the system!"

        self.maxdegree = maxdegree
        self.sklTheta = []; self.SGDTheta = []
        self.reldiff = np.zeros((\
                len(self.epochs), 
                len(self.minibatches), 
                len(self.learningrates)
                ))

        for i in range(len(self.epochs)):
            print("*"*50)
            print("epochs: ", self.epochs[i])
            for j in range(len(self.minibatches)):
                print("minibatches: ", self.minibatches[j])
                for k in range(len(self.learningrates)):
                    print("learningrate: ", self.learningrates[k])
                    for deg in tqdm(range(self.maxdegree)):
                        features = self.PolyFeatures(self.row, self.col, deg+1)
                        train, test = self.TrainTestSplit(self.height,self.testRatio)

                        heightTrain = self.height[train]; heightTest = self.height[test]
                        featuresTrain = features[train]; featuresTest = features[test]

                        scaler = StandardScaler().fit(featuresTrain)
                        featuresTrain = scaler.transform(featuresTrain); featuresTest = scaler.transform(featuresTest)
                        
                        self.sklregressor.fit(featuresTrain, heightTrain)
                        self.sklTheta.append(self.sklregressor.coef_)

                        self.regressor.fit(\
                                featuresTrain, 
                                heightTrain,
                                self.epochs[i],
                                self.minibatches[j],
                                self.learningrates[k])
                        self.SGDTheta.append(self.regressor.theta)
            
                    self.relldiff = np.zeros(self.maxdegree)

                    for i in range(self.maxdegree):
                        self.relldiff[i] = np.mean((self.sklTheta[i]-self.SGDTheta[i])**2)

