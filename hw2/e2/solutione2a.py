import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


np.random.seed(2018)
n = 50
maxdegree = 5

# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

OLSTestError = np.zeros(maxdegree)
OLSTrainError = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

nlambdas = 100
MSEPredictRidge = np.zeros(nlambdas)
lambdas = np.logspace(-4, 0, nlambdas)


for degree in range(maxdegree):
    #model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    #clf = model.fit(x_train_scaled,y_train)
    poly = PolynomialFeatures(degree=degree)
    ols = LinearRegression(fit_intercept=False)
    X_train_scaled = poly.fit(x_train_scaled)
    
    ols.fit(X_train_scaled, y_train)

    y_ols_fit = ols.predict(X_train_scaled)
    y_ols_pred = ols.predict(X_test_scaled) 
    
    for i in range(nlambdas):
        lmb = lambdas[i]
        # add ridge
        clf_ridge = skl.Ridge(alpha=lmb).fit(X_train_scaled, y_train)


    polydegree[degree] = degree
    OLSTestError[degree] = np.mean( np.mean((y_test - y_ols_pred)**2) )
    OLSTrainError[degree] = np.mean( np.mean((y_train - y_ols_fit)**2) )

plt.plot(polydegree, OLSTestError, label='Test Error')
plt.plot(polydegree, OLSTrainError, label='Train Error')
plt.legend()
plt.show()
