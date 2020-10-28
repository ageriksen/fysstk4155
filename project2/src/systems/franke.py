#!/usr/bin/env python3
import lib.regressor as reg
import lib.gradientdescent as gd

import numpy as np

class _RegressionBase:

    def __init__(self):
        self

    def PolyFeatures(self, x, y, n ):
        #Similar thoughts to the train-test split. 
        #Though somewhat more specific to models based on
        #polynomial regression, it is still aplicable to 
        #n degrees of polynomials
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
        #Pretty universal. Probably better found in
        #either an inherited class or in a separate 
        #class of "methods" or the like. to reduce 
        #repetition
        """
        takes data array and returns randomised
        set of indices for train and test splits
        """
        shuffle = np.random.permutation(data.shape[0])
        test_size = int(data.shape[0]*test_ratio)
        test = shuffle[:test_size]
        train = shuffle[test_size:]
        return train,test


class FrankeRegression(_RegressionBase):
    """
    class for performing Linear regression on the Franke function.
    I think the specific regressor(OLS, SGD, etc.) should be made 
    outside the class, as this is mainly to set up and keep the necessary 
    values, x,y,z coordinates and such. 
    """

    def __init__(self, regressor, **kwargs):
        self.regressor = regressor(**kwargs)

    def FrankeFunction(self, x,y):
        #This definitely belongs in the Franke function class. 
        #specific to the system case. 
        term1 = 0.75*np.exp(               
                -   (0.25*(9*x - 2)**2)     \
                -   0.25*((9*y - 2)**2) )

        term2 = 0.75*np.exp(            
                -   ((9*x + 1)**2)/49.0     \
                -   0.1*(9*y + 1) )

        term3 = 0.5*np.exp(              
                -   (9*x - 7)**2/4.0        \
                -    0.25*((9*y - 3)**2) )

        term4 = -0.2*np.exp(              
                -   (9*x - 4)**2            \
                -   (9*y - 7)**2 )

        return term1 + term2 + term3 + term4

    def Set(self, rows, cols, sigma): 
        ########################################
        #setup data - Should maybe be it's own method
        #And have a second "run" or "regress" method for
        #actually taking the data (perhaps with the given 
        #necessity of having set the data first, 
        #so as to run the same system over different regressors
        #e.g. OLS, Ridge, SGD, etc. 
        row = np.linspace(0,1,rows)
        col = np.linspace(0,1,cols)
        self.row_mat, self.col_mat = np.meshgrid(col, row)
        self.height_mat = self.FrankeFunction(self.row_mat, self.col_mat)
        self.row = self.row_mat.ravel()
        self.col = self.col_mat.ravel()
        self.height = self.height_mat.ravel()
        #This is mostly the specific case for Franke. 
        #Below is mainly usage of imported classes.
        ########################################

    def Run(self, testRatio=.2, maxdegree=10):

        for deg in range(1, maxdegree):
            features = self.PolyFeatures(self.row, self.col, deg)
            train, test = self.TrainTestSplit(self.height,testRatio)
            heightTrain=self.height[train]
            heightTest=self.height[test]
            featuresTrain=features[train]
            featuresTest=features[test]

            SGDbeta = self.regressor.fit(featuresTrain, heightTrain)
            SGDpredTrain = featuresTrain@SGDbeta
            SGDpredTest = featuresTest@SGDbeta
