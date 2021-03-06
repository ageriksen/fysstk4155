#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer

from IPython.display import display 
from pylab import plt, mpl


plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

np.random.seed(2020)

cancer = load_breast_cancer()

cdf = pd.DataFrame(np.c_[cancer.data, cancer.target], 
        columns=np.append(cancer.feature_names, ['target']) )

display(cdf)

Xtrain, Xtest, ytrain, ytest = train_test_split(cancer.data, cancer.target)
print(Xtrain.shape)
print(Xtest.shape)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(Xtrain, ytrain)
print('test set accuracy with logreg: {:.2f}'.format(logreg.score(Xtest, ytest)))

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain_scaled = scaler.transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

logreg.fit(Xtrain_scaled, ytrain)
print('test set accuracy with logreg, scaled data: {:.2f}'.format(logreg.score(Xtest_scaled, ytest)))

correlation = cdf.drop(['target'], axis=1).corrwith(cdf['target'])
important = correlation.abs().sort_values(ascending=False)[:int(correlation.size*.3)]
print('*'*10)
print(important)
print('*'*10)

Xtrain, Xtest, ytrain, ytest = train_test_split(cdf[important.index], cdf['target'])

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain_scaled = scaler.transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

logreg.fit(Xtrain_scaled, ytrain)
print('test accuracy, logreg, scaled data, important inputs: {:.2f}'.format(logreg.score(Xtest_scaled, ytest)))
#TODO reduce nr. of datapoints when fitting to encourage overfitting -> see if "best features" has more to say then.
#TODO implement resampling methods.
