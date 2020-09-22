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

#=====================OLS
beta = np.linalg.inv( X_train.T @ X_train) @ X_train.T @ y_train
print("====================")
print("OLS\n")
print("manual model")
print("optimal beta\n", beta)
y_tilde = X_train @ beta
y_pred = X_test @ beta

print("R2, training")
print("{:.2f}".format(R2(y_train, y_tilde)))
print("MSE, training")
print("{:.2f}".format(MSE(y_train, y_tilde)))
print("----------------")
print("R2, test")
print("{:.2f}".format(R2(y_test, y_pred)))
print("MSE, test")
print("{:.2f}".format(MSE(y_test, y_pred)))
print("====================", flush=True)
#=====================!OLS

#identity matrix, dimensions of featuresxfeatures
_I = np.eye(p,p)

nlambdas = 100
MSEtest  = np.zeros(nlambdas)
MSE_skl  = np.zeros(nlambdas)
MSEtrain = np.zeros(nlambdas)
lambdas = np.logspace(-4, 0, nlambdas)

for i in range(nlambdas):
    lmbd = lambdas[i]
    
    beta_ridge = np.linalg.inv(X_train.T@X_train + lmbd*_I)@X_train.T@y_train
    ytilde_ridge = X_train @ beta_ridge
    ypred_ridge = X_test @ beta_ridge
    MSEtrain[i] = MSE(y_train, ytilde_ridge)
    MSEtest[i] = MSE(y_test, ypred_ridge)

    
    ridge  = skl.Ridge(alpha=lmbd).fit(X_train, y_train)
    y_skl = ridge.predict(X_test)
    MSE_skl[i] = MSE(y_test, y_skl)

plt.plot(np.log10(lambdas), MSEtrain, label="train")
plt.plot(np.log10(lambdas), MSEtest, '--', label="test")
plt.plot(np.log10(lambdas), MSE_skl, '--', label="skl")
plt.legend()
plt.xlabel("$log_{10}(\lambda$)")
plt.ylabel("MSE")
plt.show()
"""
There is something going on here. The self made code reports very small errors not seen in skl
There also seems to take a little longer to do the skl version. This should be mostly opposite
of what you'd expect. I imagine there are errors I am not accounting for with my code, as well
as the pipelines and QA in skl slows down the setup for smaller sets like this. I imagine any larger
versions would be quicker. I trust the skl version more than my sloppy code. 

After some testing and comparing to the skl in the solution given, turns out there was a difference in 
the way I calculated the skl ridge and the solution.
the way I set up:
==================
    ridge = skl.Ridge(alpha = lmbd)
    ridge.fit(X_train, y_train[:, np.newaxis]) #newaxis adds 1 dim to array. sklearn needs the dimension
    y_skl = ridge.predict(X_test)
    MSE_skl[i] = MSE(y_test, y_skl)
==================
the solution setup:
==================
    ridge  = skl.Ridge(alpha=lmbd).fit(X_train, y_train)
    y_skl = ridge.predict(X_test)
    MSE_skl[i] = MSE(y_test, y_skl)
==================
seems the newaxis messed it up. Perhaps I misremembered something abt the axes required?
might have to do with the intercept? 

"""
