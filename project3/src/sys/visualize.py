#!/usr/bin/env python3
"""
Found in ProjectsData/Project2_1.py in the course 
repo. 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

import pandas as pd
import seaborn as sns

np.random.seed(2020)

cancer = load_breast_cancer()

inputs = cancer.data
outputs = cancer.target # malignant/benign
labels = cancer.feature_names[0:30]

n_inputs = len(inputs)

print("*"*10)
print("breast cancer dataset:")
print("inputs:\n", inputs)
print("outputs:\n", outputs)
print("labels:\n", labels)
print("number of inputs: ", n_inputs)
print("*"*10)


meanpd = pd.DataFrame(inputs, columns=labels)
corr = meanpd.corr().round(1) # pairwise correlation collumns, exclude NA/null values
        
#plot heatmap of correlation matrix
plt.figure()
sns.heatmap(corr, cbar=True, 
        xticklabels=labels, yticklabels=labels,
        cmap='YlOrRd')
plt.show()

