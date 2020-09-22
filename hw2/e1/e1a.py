
import numpy as np
import matplotlib.pyplot as plt

#Could probably write functionality for this, but it is unlikely to be worth it 
#at this stage
from sklearn.preprocessing import StandardScaler

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
noise = .7
y = 2. + 5*x**2 + noise*np.random.randn(100)

#feature matrix, here with p features
p = 3
X = np.zeros((len(x), p))
X[:,0] = 1
X[:,1] = x
X[:,2] = x**2

X_train, X_test, y_train, y_test = split_data(X, y)

#scaling the data
#scaler = StandardScaler()
#scaler.fit(X_train) 
#X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)

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
MSEtrain = np.zeros(nlambdas)
MSEtest = np.zeros(nlambdas)
lambdas = np.logspace(-4, 2, nlambdas)
for i in range(nlambdas):
    lmbd = lambdas[i]
    beta_ridge = np.linalg.inv(X_train.T@X_train + lmbd*_I)@X_train.T@y_train
    ytilde_ridge = X_train @ beta_ridge
    ypred_ridge = X_test @ beta_ridge
    MSEtrain[i] = MSE(y_train, ytilde_ridge)
    MSEtest[i] = MSE(y_test, ypred_ridge)

plt.plot(lambdas, MSEtrain, label="train")
plt.plot(lambdas, MSEtest, label="test")
plt.legend()
plt.xlabel("$log_{10}(\lambda$)")
plt.ylabel("MSE")
"""
The inclusion of the varying lambda allows us to find improved fits for the actual test, 
while the training is somewhat less accurate. 
There is a pretty good overlap for the minima of train and test MSE for Ridge compared to the 
OLS, which essentially pertains to the lambda=0 stage

"""
plt.show()
