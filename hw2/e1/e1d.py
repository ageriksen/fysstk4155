import numpy as np
import matplotlib.pyplot as plt

import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#setting seed
np.random.seed(3155)

#functions
def split_data(data, target, test_ratio=0.2):
    #Split the inputs and outputs into training and test versions
    indices = np.random.permutation(data.shape[0])
    test_set_size = int(data.shape[0]*test_ratio)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return data[train_indices], data[test_indices], target[train_indices], target[test_indices]

def R2(y, y_tilde):
    return 1 - np.sum( (y - y_tilde)**2 )/np.sum( (y - np.mean(y))**2 )

def MSE(y, y_tilde):
    n = np.size(y)
    return np.sum( (y - y_tilde)**2 )/n

#Generating data points, 100 pts in x and y
x = np.random.rand(100)
y = 2.0+5*x*x+0.1*np.random.randn(100)

# number of features p (here degree of polynomial
p = 3
#  The design matrix now as function of a given polynomial
X = np.zeros((len(x),p))
X[:,0] = 1.0
X[:,1] = x
X[:,2] = x*x

#X_train, X_test, y_train, y_test = split_data(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#identity matrix, dimensions of featuresxfeatures
_I = np.eye(p,p)

nlambdas = 1000
MSEridge_test  = np.zeros(nlambdas)
MSEridge_train = np.zeros(nlambdas)

MSEridge_skl  = np.zeros(nlambdas)

MSElasso_skl  = np.zeros(nlambdas)

lambdas = np.logspace(-4, 3, nlambdas)

for i in range(nlambdas):
    lmbd = lambdas[i]
    
    beta_ridge = np.linalg.inv(X_train.T@X_train + lmbd*_I)@X_train.T@y_train
    ytilde_ridge = X_train @ beta_ridge
    ypred_ridge = X_test @ beta_ridge
    MSEridge_train[i] = MSE(y_train, ytilde_ridge)
    MSEridge_test[i] = MSE(y_test, ypred_ridge)

    
    ridge  = skl.Ridge(alpha=lmbd).fit(X_train, y_train)
    yridge_skl = ridge.predict(X_test)
    MSEridge_skl[i] = MSE(y_test, yridge_skl)

    lasso = skl.Lasso(alpha=lmbd).fit(X_train, y_train)
    ylasso_skl = lasso.predict(X_test)
    MSElasso_skl[i] = MSE(y_test, ylasso_skl)

plt.plot(np.log10(lambdas), MSEridge_train, label="ridge train")
plt.plot(np.log10(lambdas), MSEridge_test, '--', label="ridge test")
plt.plot(np.log10(lambdas), MSEridge_skl, '--', label="ridge skl")
plt.plot(np.log10(lambdas), MSElasso_skl, '--', label="lasso skl")
plt.legend()
plt.xlabel("$log_{10}(\lambda$)")
plt.ylabel("MSE")
plt.show()
