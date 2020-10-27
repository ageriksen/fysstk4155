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

    def __init__(self):
        self

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


    def RunFranke(self, rows, cols, sigma):
        ########################################
        #setup data - Should maybe be it's own method
        #And have a second "run" or "regress" method for
        #actually taking the data (perhaps with the given 
        #necessity of having set the data first, 
        #so as to run the same system over different regressors
        #e.g. OLS, Ridge, SGD, etc. 
        row = np.linspace(0,1,rows)
        col = np.linspace(0,1,cols)
        row_mat, col_mat = np.meshgrid(col, row)
        height_mat = self.FrankeFunction(row_mat, col_mat)
        row = row_mat.ravel()
        col = col_mat.ravel()
        height = height_mat.ravel()
        #This is mostly the specific case for Franke. 
        #Below is mainly usage of imported classes.
        ########################################

        #The polynomial degree should be varied. If possible,
        #probably easiest to have the model complexity in this
        #section and input the eventual other parameters in the
        #regressor case, either through passing **kwargs or *args
        #to generalize the process. 

        maxdegree = 10
        for deg in range(maxdegree):
            features = self.PolyFeatures(row, col, deg)
            testRatio = .2
            train, test = self.TrainTestSplit(height,testRatio)
            heightTrain=height[train]
            heightTest=height[test]
            featuresTrain=features[train]
            featuresTest=features[test]

            ols = reg.OLS()   
            olsbeta = ols.fit(featuresTrain, heightTrain)
            olspredTrain = ols.predict(featuresTrain)
            olspredTest = ols.predict(featuresTest)

            #learningrate = 0.001 #should be input somewhere in the call IF it's GD. w/ SGD, use learning schedule.
            minibatches = 10
            epochs = 50
            gradient = gd.SGD(minibatches, epochs)
            SGbeta = gradient.FindBeta(featuresTrain, heightTrain)
            SGpredTrain = featuresTrain@SGbeta
            SGpredTest = featuresTest@SGbeta

            #print("OLS beta, train, test:\n",olsbeta,"\n", olspredTrain,"\n", olspredTest)
            #print("*"*25)
            #print("SGD beta, train, test:\n",SGbeta,"\n", SGpredTrain,"\n", SGpredTest)
            #print("*"*25)
            #print("difference beta\n",olsbeta - SGbeta)
            #print("*"*25)
