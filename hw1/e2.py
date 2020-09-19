
import numpy as np
import matplotlib.pyplot as plt
#sklearn imports:
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#solution imports below
import pandas as pd
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#

def split_data(data, target, test_ratio=0.2):
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

#x = np.random.rand(100, 1)
x = np.random.rand(100)
#y = 2.0+5*x**2 + 0.1*np.random.randn(100,1)
y = 2.0+5*x**2 + 0.1*np.random.randn(100)

#To compute a 2nd order polynomial fit, I'd need a 3 feature feature matrix
X = np.zeros((len(x), 3))
X[:,0] = 1
X[:,1] = x
X[:,2] = x**2

X_train, X_test, y_train, y_test = split_data(X, y)

beta = np.linalg.inv( X_train.T @ X_train) @ X_train.T @ y_train
print("====================")
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
print("====================")



"""
implementing sklearn to compare
"""
clf = skl.LinearRegression().fit(X_train, y_train)
y_predskl = clf.predict(X_test)

print("sklearn model")
print("R2 = {:.2f}".format(clf.score(X_test, y_test)))
print("MSE = {:.2f}".format(mean_squared_error(y_predskl, y_test)))
print("====================")


plt.plot(y_test, 'o', color='orange', label='output')
plt.plot(y_pred, 'o', label='manual model')
plt.plot(y_predskl, '+', label='skl model')
plt.legend()
plt.show()

"""
for sklearn method:
"""
clf = skl.LinearRegression().fit(X, y)
ytilde = clf.predict(X)
print("MSE: %.2f" % mean_squared_error(y, ytilde))
print("R2 score: %.2f" % r2_score(y, ytilde))
print(clf.coef_, clf.intercept_)
