import os
import numpy as np
import csv

#import pandas as pd



datadir = "/home/ms/uni/fys-stk4155/coursegit/doc/pub/Regression/ipynb/DataFiles/"
datafile = "EoS.csv"

if( os.path.exists(datadir) ):
    datapath = os.path.join(datadir, datafile)
else:
    print("-----------\ncouldn't find datafile ", datafile, " in path\n", datadir, "\n-----------")
    quit()
#data = pd.read_fwf(infile, names=('col1', 'col2'))

       

col1 = []
col2 = []
for row in csv.reader(open(datapath), delimiter=','):
    col1.append(float(row[0]))
    col2.append(float(row[1]))

col1 = np.asarray(col1)
col2 = np.asarray(col2)

print("there are ", len(col1), " datapoints with 2 features per point in the file.")
"""
runresult:
there are  90  datapoints with 2 features per point in the file.
"""
