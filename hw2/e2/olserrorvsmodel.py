import numpy as np
import matplotlib.pyplot as plt

#import sklearn.linear_model as skl
from sklearn.linear_model       import LinearRegression
from sklearn.preprocessing      import PolynomialFeatures
from sklearn.model_selection    import train_test_split
from sklearn.pipeline           import make_pipeline 


np.random.seed(2020)
n = 100
maxdegree = 14
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

# split in training and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

for deg in range(maxdegree):
    #make_pipeline(X, fit_method)?
    pipeline = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression(fit_intercept=False))
    clf = pipeline.fit(x_train, y_train)
