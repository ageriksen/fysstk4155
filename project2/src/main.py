#!/usr/bin/env python3

import lib.regressor as reg
import lib.gradientdescent as gd

import numpy as np

def main():
    rows = 100; cols = 100; sigma = 0.1
    RunFranke(rows, cols, sigma)

def FrankeFunction(x,y):
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

def PolyFeatures(x, y, n ):
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

def TrainTestSplit(data, test_ratio=.2):
    """
    takes data array and returns randomised
    set of indices for train and test splits
    """
    shuffle = np.random.permutation(data.shape[0])
    test_size = int(data.shape[0]*test_ratio)
    test = shuffle[:test_size]
    train = shuffle[test_size:]
    return train,test

def RunFranke(rows, cols, sigma):
    #setup data
    row = np.linspace(0,1,rows)
    col = np.linspace(0,1,cols)
    row_mat, col_mat = np.meshgrid(col, row)
    height_mat = FrankeFunction(row_mat, col_mat)
    row = row_mat.ravel()
    col = col_mat.ravel()
    height = height_mat.ravel()

    degree = 5
    features = PolyFeatures(row, col, degree)
    testRatio = .2
    train, test = TrainTestSplit(height,testRatio)
    heightTrain=height[train]
    heightTest=height[test]
    featuresTrain=features[train]
    featuresTest=features[test]

    beta = np.linalg.pinv(featuresTrain.T @ featuresTrain)\
            @ (featuresTrain.T @ heightTrain)

    predictTrain = featuresTrain@beta
    predictTest = featuresTest@beta

    ols = reg.OLS()   
    OObeta = ols.fit(featuresTrain, heightTrain)
    OOpredTrain = ols.predict(featuresTrain)
    OOpredTest = ols.predict(featuresTest)

    learningrate = 0.001
    gradient = gd.SGD(learningrate)
    SGbeta = gradient.FindBeta(featuresTrain, heightTrain)
    SGpredTrain = featuresTrain@SGbeta
    SGpredTest = featuresTest@SGbeta

    print("OLS beta, train, test:")
    print(beta,"\n", predictTrain,"\n", predictTest)
    print("*"*25)
    



if __name__ == "__main__":
    main()
