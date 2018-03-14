import matplotlib.pylab as plt
import numpy as np
from pygfl.easy import solve_gfl

truth = np.zeros(200)
truth[:50] = 0.5
truth[50:100] = 0.75
truth[100:150] = 0.25
truth[150:180] = 0.1
truth[180:] = 0.9

trials = np.random.poisson(10, size=200)
successes = np.array([(np.random.random(size=t) <= p).sum() for t,p in zip(trials, truth)])

beta = solve_gfl((trials, successes), loss='binomial')

plt.scatter(np.arange(200)+1, successes / trials.astype(float))
plt.plot(np.arange(200)+1, truth, color='gray', alpha=0.5)
plt.plot(np.arange(200)+1, 1. / (1+np.exp(-beta)), color='orange')
plt.show()
