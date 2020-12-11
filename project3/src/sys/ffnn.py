#!/usr/bin/env python3
import numpy as np
import pandas as pd
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


import torch
import torch.nn.functional as F

torch.manual_seed(2020)
X = torch.tensor(
        torch.unsqueeze(torch.from_numpy(cancer.data), dim=1),
        dtype=torch.float32)
y = torch.from_numpy(cancer.target)
y = y.reshape(y.numel(), 1)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)#hiddenlayer
        self.predict = torch.nn.Linear(n_hidden, n_output)#outputlayer

    def forward(self, x):
        x = F.relu(self.hidden(x))#activation for hidden layer
        x = self.predict(x) #linear output
        return x

#net = Net(n_feature=len(cancer.feature_names), n_hidden=len(cancer.feature_names), n_output=len(cancer.feature_names))
net = Net(n_feature=len(cancer.feature_names), n_hidden=10, n_output=1)
#print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=.2)
loss_func = torch.nn.CrossEntropyLoss()#TODO find the correct loss func for the nn w/ binary classification

epochs = 200
for t in range(epochs):

    prediction = net(X) #predict based on x

    loss = loss_func(prediction, y)#( 1: nn output, 2: target )

    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    

