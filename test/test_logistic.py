import matplotlib.pylab as plt
import numpy as np
from pygfl.easy import solve_gfl

truth = np.zeros(200)
truth[:50] = 0.5
truth[50:100] = 0.75
truth[100:150] = 0.25
truth[150:180] = 0.1
truth[180:] = 0.9

data = (np.random.random(size=200) <= truth).astype(int)

beta = solve_gfl(data, loss='logistic')

plt.scatter(np.arange(200)+1, data)
plt.plot(np.arange(200)+1, truth, color='gray', alpha=0.5)
plt.plot(np.arange(200)+1, 1. / (1+np.exp(-beta)), color='orange')
plt.show()
