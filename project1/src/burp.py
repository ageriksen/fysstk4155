from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from random import random, seed
np.random.seed(2020)

#Functions
def FrankeFunction(x,y):
    term1 = 0.75*np.exp( -(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2) )
    term2 = 0.75*np.exp( -((9*x + 1)**2)/49.0 - 0.1*(9*y + 1) )
    term3 = 0.5*np.exp( -(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2) )
    term4 = -0.2*np.exp( -(9*x - 4)**2 - (9*y - 7)**2 )
    return term1 + term2 + term3 + term4

def create_X(x, y, n ):
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

# Make data.
steplength = 0.05
x = np.arange(0, 1, steplength)
y = np.arange(0, 1, steplength)

row_mat, col_mat = np.meshgrid(x,y)
z = FrankeFunction(row_mat, col_mat)

#Create figures
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
# Plot the surface.
surf = ax1.plot_surface(row_mat, col_mat, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax1.set_zlim(-0.10, 1.40)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig1.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

##########################

row_arr = row_mat.ravel()
col_arr = col_mat.ravel()
z_arr = z.ravel()
maxdeg = 5
X = create_X(row_arr, col_arr, maxdeg)
X_train, X_test, z_train, z_test = split_data(X,z_arr)# z.ravel())
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
z_tilde = X_train @ beta
z_pred = X_test @ beta

MSE_train = MSE(z_train, z_tilde)
MSE_test = MSE(z_test, z_pred)
#print("MSE_train:   ", MSE_train)
#print("MSE_test:    ", MSE_test)
#
#
#fig2 = plt.figure()
#ax2 = fig2.gca(projection='3d')
## Plot the surface.
#surf = ax2.plot_surface(x, y, z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#
## Customize the z axis.
#ax2.set_zlim(-0.10, 1.40)
#ax2.zaxis.set_major_locator(LinearLocator(10))
#ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
## Add a color bar which maps values to colors.
#fig2.colorbar(surf, shrink=0.5, aspect=5)
#
#plt.show()
