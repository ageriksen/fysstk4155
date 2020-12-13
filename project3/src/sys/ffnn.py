#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

from IPython.display import display 
from pylab import plt, mpl


import torch
import torch.nn.functional as F

##################################################
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

np.random.seed(2020)
torch.manual_seed(2020)

##################################################

cancer = load_breast_cancer()

cdf = pd.DataFrame(np.c_[cancer.data, cancer.target], 
        columns=np.append(cancer.feature_names, ['target']) )

display(cdf)
##################################################


X = torch.tensor(cancer.data, dtype=torch.float32)
y = torch.tensor(cancer.target, dtype=torch.float32)
X = X.reshape(y.numel(), len(cancer.feature_names))
y = y.reshape(y.numel(), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)#hiddenlayer
        self.predict = torch.nn.Linear(n_hidden, n_output)#outputlayer
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.hidden(x))#activation for hidden layer
        x = self.predict(x) #linear output
        x = self.sigmoid(x)
        return x

#net = Net(n_feature=len(cancer.feature_names), n_hidden=len(cancer.feature_names), n_output=len(cancer.feature_names))
net = Net(n_feature=len(cancer.feature_names), n_hidden=10, n_output=1)
#print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=.2)
loss_func = torch.nn.BCELoss()

epochs = 1000
accuracy_train = np.zeros(epochs)
accuracy_test = np.zeros(epochs)
#stop = epochs
count = 0
for t in range(epochs):

    prediction_train = net(X_train_scaled) #predict based on x
    prediction_test = net(X_test_scaled)

    loss = loss_func(prediction_train, y_train)#( 1: nn output, 2: target )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy_train[t] = 1 - torch.mean(abs( torch.round(prediction_train) - y_train ) )
    accuracy_test[t] = 1 - torch.mean(abs( torch.round(prediction_test) - y_test ) )

    if accuracy_test[t] > accuracy_test[t-1]:
        best = accuracy_test[t]
        count = 0
    if accuracy_test[t] < best:
        count += 1
        if count > 25: # 10 epochs beyond the last increase in accuracy
            stop = t
            break


accuracy_train = accuracy_train[:stop]
accuracy_test = accuracy_test[:stop]

#print(accuracy_train)
plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.show()
