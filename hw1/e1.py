import os
import numpy as np
import csv

#import pandas as pd


def split_data(data, target, test_ratio=0.2):
    """ 
    Input is the data/input and the target/output as well as an optional ratio for test 
    and training data. The permutation numpy option randomizes the order of the range given
    and serve as the indices for the slices of our domain we return
    """
    shuffled_indices = np.random.permutation(data.shape[0])
    test_set_size = int(data.shape[0]*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices], target[train_indices], target[test_indices]

def R2(y, y_tilde):
    return 1 - ( np.sum( (y-y_tilde)**2 )/np.sum( (y-np.mean(y))**2 ) )

def MSE(y, y_tilde): 
    n = np.size(y)
    return np.sum( (y-y_tilde)**2 )/n


#datadir = "/home/ms/uni/fys-stk4155/coursegit/doc/pub/Regression/ipynb/DataFiles/"
datadir = "../../coursegit/doc/pub/Regression/ipynb/DataFiles/"
datafile = "EoS.csv"

if( os.path.exists(datadir) ):
    datapath = os.path.join(datadir, datafile)
else:
    print("-----------\ncouldn't find datafile ", datafile, " in path\n", datadir, "\n-----------")
    quit()
#data = pd.read_fwf(infile, names=('densities', 'energies'))

       

densities = [] #density
energies = [] #energy
for row in csv.reader(open(datapath), delimiter=','):
    densities.append(float(row[0]))
    energies.append(float(row[1]))

densities = np.asarray(densities)
energies = np.asarray(energies)

print("there are ", len(densities), " datapoints with 2 features per point in the file.")
"""
runresult:
there are  90  datapoints with 2 features per point in the file.
"""

"""
We want to model the data through a 3rd degree polynomial, so we need a 4-collumn design matrix. 
We want to then apply the densities to the model according to some polynomial. For the equation of 
state, there have been such polynomials constructed, e.g. the liquid drop model for based on masses. 
For an equation of state using densities, there are other polynomials. For simplicity, using the version 
utilized in the solution.
"""
X = np.zeros((len(densities), 5))
X[:,0] = 1
X[:,1] = densities**(2./3.)
X[:,2] = densities
X[:,3] = densities**(4./3.)
X[:,4] = densities**(5./3.)
#print(X)

X_train, X_test, y_train, y_test = split_data(X, energies)

Beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train # = (X^T X)^-1 X^T y

#Training
y_tilde = X_train @ Beta
print("R2 training")
print(R2(y_train, y_tilde))
print("MSE training")
print(MSE(y_train, y_tilde))

y_pred = X_test @ Beta
print("R2 test")
print(R2(y_test, y_pred))
print("MSE test")
print(MSE(y_test, y_pred))
"""
R2 training
0.9999845049202144
MSE training
6.8179304148952244
R2 test
0.9999875787440466
MSE test
4.676469383209456
"""

