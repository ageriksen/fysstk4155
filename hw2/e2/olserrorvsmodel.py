import numpy as np
import matplotlib.pyplot as plt

#import sklearn.linear_model as skl
from sklearn.linear_model       import LinearRegression
from sklearn.preprocessing      import PolynomialFeatures, StandardScaler
from sklearn.model_selection    import train_test_split
from sklearn.pipeline           import make_pipeline 
#from sklearn.metrics            import r2_score, mean_squared_error


np.random.seed(2020)
n = 1000
maxdegree = 100
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)

# split in training and test data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

#R2s = np.zeros(maxdegree)
#MSEs = np.zeros(maxdegree) 
trainerror = np.zeros(maxdegree)
testerror = np.zeros(maxdegree)
modeldeg = np.arange(maxdegree)

for deg in range(maxdegree):
    #make_pipeline(X, fit_method)?
    pipeline = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression(fit_intercept=False))
    clf = pipeline.fit(x_train_scaled, y_train)
    y_tilde = clf.predict(x_train_scaled)
    y_pred = clf.predict(x_test_scaled)
    #R2s[deg] = r2_score(y_test, y_predict)
    #MSEs[deg] = mean_squared_error(y_test, y_predict)
    trainerror[deg] = np.mean( np.mean( (y_train - y_tilde)**2) ) 
    testerror[deg] = np.mean( np.mean( (y_test - y_pred)**2) ) 

    

#plt.plot(modeldeg, R2s, label='R2 score')
#plt.plot(modeldeg, MSEs, label='MSE')
plt.plot(modeldeg, trainerror, label='train error')
plt.plot(modeldeg, testerror, label='test error')
plt.legend()
plt.show()
