#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
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

#inputs = cancer.data
#outputs = cancer.target # malignant/benign
#labels = cancer.feature_names[0:30]
#
#inputpd = pd.DataFrame(inputs, columns=labels)
#outputpd = pd.Series(outputs)
