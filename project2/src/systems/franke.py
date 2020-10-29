#!/usr/bin/env python3
import lib.regressor as reg
import lib.gradientdescent as gd

import sklearn.linear_model as skl

import numpy as np
from tqdm import tqdm

class _FrankeBaseRegressor:

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

    def Set(self, rows, cols, sigma): 
        row = np.linspace(0,1,rows)
        col = np.linspace(0,1,cols)
        self.row_mat, self.col_mat = np.meshgrid(col, row)
        self.height_mat = self.FrankeFunction(self.row_mat, self.col_mat)
        self.row = self.row_mat.ravel()
        self.col = self.col_mat.ravel()
        self.height = self.height_mat.ravel()
        self.setup = True


class FrankeRegression(_FrankeBaseRegressor):
    """
    class for performing Linear regression on the Franke function.
    I think the specific regressor(OLS, SGD, etc.) should be made 
    outside the class, as this is mainly to set up and keep the necessary 
    values, x,y,z coordinates and such. 
    """

    def __init__(self, regressor, **kwargs):
        self.regressor = regressor(**kwargs)

    def Run(self,  testRatio=.2, maxdegree=10):
        assert self.setup, "You need to setup the system!"

        regtrain = []
        regtest = []
        skltrain = []
        skltest = []
        sklR2 = []
        sklMSE = []
        
        for deg in tqdm(range(1, maxdegree)):

            features = self.PolyFeatures(self.row, self.col, deg)
            train, test = self.TrainTestSplit(self.height,testRatio)
            heightTrain=self.height[train]
            heightTest=self.height[test]
            featuresTrain=features[train]
            featuresTest=features[test]

            sklmodel = skl.LinearRegression()
            sklmodel.fit(featuresTrain, heightTrain)
            sklmodel.predict(featuresTrain)
            skltrain.append(sklmodel.predict(featuresTrain))
            skltest.append(sklmodel.predict(featuresTest))
            sklR2.append(sklmodel.score(heightTest, sklmodel.predict(featuresTest)))

            self.regressor.fit(featuresTrain, heightTrain)
            regtrain.append(featuresTrain@self.regressor.theta)
            print(type(featuresTrain@self.regressor.theta), flush=True)
            regtest.append(featuresTrain@self.regressor.theta)

        
