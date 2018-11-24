#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

test_labels = np.loadtxt('../data/test_labels.dat')
test_predictions = np.loadtxt('predictions.dat') 

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
plt.show()
