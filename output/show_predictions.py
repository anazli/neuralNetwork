#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

labels = np.loadtxt('../data/boston_house/test_labels.dat')
pred = np.loadtxt('predictions.dat')

error = pred - labels

plt.figure()
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.show()
