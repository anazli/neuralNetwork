#!/usr/bin/env python3

import numpy as np
import pandas as pd

df = pd.read_csv('letter-recognition.data', header=None)
d =  {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8,'J':
       9, 'K':10, 'L':11, 'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17, 'S':
       18, 'T':19, 'U':20, 'V':21, 'W':22, 'X':23, 'Y':24,'Z':25}

letters = np.array(df[0])
features = np.array(df.iloc[:,1:])

labels = np.zeros([letters.shape[0]]) 
for i in range(len(letters)):
    labels[i] = d[letters[i]]

data_size = labels.shape[0]
train_size = 16000
test_size = data_size - train_size


np.savetxt('train_data.dat', features[:train_size,:], fmt='%d')
np.savetxt('train_labels.dat', labels[:train_size], fmt='%d')
np.savetxt('test_data.dat', features[train_size:,:], fmt='%d')
np.savetxt('test_labels.dat', labels[train_size:], fmt='%d')


