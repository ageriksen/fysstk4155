import numpy as np
import matplotlib.pyplot as plt

import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#setting seed
np.random.seed(3155)

#functions
def MSE(y, y_tilde):
    n = np.size(y)
    return np.sum( (y - y_tilde)**2 )/n

#Generating data points, 100 pts in x and y
x = np.random.rand(100)
noise = .7
y = 2. + 5*x**2 + noise*np.random.randn(100)

#feature matrix, here with p features
p = 3
X = np.zeros((len(x), p))
X[:,0] = 1
X[:,1] = x
X[:,2] = x**2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


nlambdas = 100
lambdas = np.logspace(-4, 2, nlambdas)

MSEtrain_ridge = np.zeros(nlambdas)
MSEtest_ridge = np.zeros(nlambdas)

for i in range(nlambdas):
    lmbd = lambdas[i]
    ridge = skl.Ridge(alpha = lmbd)
    ridge.fit(X_train, y_train[:, np.newaxis]) #newaxis adds 1 dim to array. sklearn needs the dimension
    
    y_tilde = ridge.predict(X_train)
    y_pred = ridge.predict(X_test)
#    MSEtrain_ridge[i] = mean_squared_error(y_train, y_tilde)
#    MSEtest_ridge[i] = mean_squared_error(y_test, y_pred)
    MSEtrain_ridge[i] = MSE(y_train, y_tilde)
    MSEtest_ridge[i] = MSE(y_test, y_pred)

plt.plot(lambdas, MSEtrain_ridge, label='train')
plt.plot(lambdas, MSEtest_ridge, label='test')
plt.xlabel("$log_{10}(\lambda)$")
plt.ylabel("MSE")
plt.legend()
plt.show()
