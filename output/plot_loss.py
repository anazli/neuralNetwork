#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt('loss.dat')
plt.figure()
plt.plot(y)
plt.show()