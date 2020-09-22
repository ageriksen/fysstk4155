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

X_train, X_test, y_train, y_test = split_data(X, y)

inverse = np.linalg.inv( X_train.T @ X_train )
beta_ols = inverse @ X_train.T @ y_train
print("OLS beta: ", beta_ols)
print("(X^T X)^-1: \n", np.linalg.inv( X_train.T @ X_train ))
#variances = np.zeros(p)
#[variances[i]=inverse[i,i] for i in range(p)]
variances = np.array([inverse[i,i] for i in range(inverse.shape[0])])
#print("diagonal elements (variances):\n", variances)
#print("mean variance: ", np.mean(variances))


I = np.eye(p,p)
nlambdas = 100
lambdas = np.logspace(-4, 2, nlambdas)
variances = np.zeros((nlambdas, p))
meanVar = np.zeros(nlambdas)
#print("=======Ridge:=======")
for i in range(nlambdas):
    lmbd = lambdas[i]
    inverse = np.linalg.inv( X_train.T @ X_train + lmbd*I)
    beta_ridge = inverse @ X_train.T @ y_train
    variances[i,:] = np.array([inverse[j,j] for j in range(p)])
    meanVar[i] = np.mean(variances[i,:])
    #print("variances, lambda={:.5f}\n".format(lmbd), variances[i,:])
    #print("mean variance: ", meanVar[i] , "\n")


plt.plot(np.log10(lambdas), meanVar)
plt.xlabel("log10(lambdas)")
plt.ylabel("$\mathbb{E}[Var]$")
plt.show()
