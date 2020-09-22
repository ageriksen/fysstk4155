import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


np.random.seed(2020)
n = 100
maxdegree = 5
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

#X = np.zeros((len(x), maxdegree))
#for deg in range(maxdegree):
#    X[:,deg] = x**deg

poly = PolynomialFeatures(degree=maxdegree)

# split in training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
