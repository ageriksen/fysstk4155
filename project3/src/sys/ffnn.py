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
#X = torch.tensor(
#        torch.squeeze(torch.from_numpy(cancer.data), dim=1),
#        dtype=torch.float32)
#y = torch.from_numpy(cancer.target, dtype=torch.float32)
#y = y.reshape(y.numel(), 1)

X = torch.tensor(cancer.data, dtype=torch.float32)
y = torch.tensor(cancer.target, dtype=torch.float32)

X = X.reshape(y.numel(), len(cancer.feature_names))
y = y.reshape(y.numel(), 1)

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

epochs = 10
difference = np.zeros(epochs)
for t in range(epochs):

    #print('starting prediction')
    prediction = net(X) #predict based on x

    #print('finding loss')
    loss = loss_func(prediction, y)#( 1: nn output, 2: target )

    #print('optimizing')
    optimizer.zero_grad()
    #print('backward')
    loss.backward()
    #print('step optimizer')
    optimizer.step()
    #print("this is where I'd find the score")
    difference[t] = torch.mean(abs( torch.round(prediction) - y ) )
    #print("score found")


print(difference)
#plt.plot(difference)
#plt.show()
