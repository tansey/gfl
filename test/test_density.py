import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp1d
from density import GraphFusedDensity
from collections import defaultdict


if __name__ == '__main__':
    bins = []
    data = np.array([[1,1,3,5,2,1,5,10,8,10,11,1,1,1,2,1],
                    [1,1,2,6,3,1,5,6,2,2,1,1,1,1,2,1]], dtype='int32')

    ntrails = 1
    trails = np.arange(data.shape[0], dtype='int32')
    breakpoints = np.array([data.shape[0]], dtype='int32')
    edges = defaultdict(list)
    for x,y in zip(np.arange(data.shape[0]-1), np.arange(1,data.shape[0])):
        edges[x].append(y)
        edges[y].append(x)


    gfd = GraphFusedDensity(polya_levels=3)
    gfd.set_data(data, edges, ntrails, trails, breakpoints)

    print 'Data:'
    for row in data:
        print '[{0}] total={1}'.format(','.join([str(x) for x in row]), row.sum())
    for j, (left, mid, right, trials, successes) in enumerate(gfd.bins):
        print '\tBin #{0} [{1},{2},{3}] N={4} K={5}'.format(j, left, mid, right, trials, successes)

    
    results = gfd.solution_path()


    raw = data / data.sum(axis=1, dtype=float)[:,np.newaxis]
    fig, ax = plt.subplots(2)
    ax[0].plot(np.arange(data.shape[1]), raw[0], marker='o', label='MLE')
    ax[0].plot(np.arange(data.shape[1]), results['aic_densities'][0], label='Smoothed', color='orange')
    ax[1].plot(np.arange(data.shape[1]), raw[1], marker='o', label='MLE')
    ax[1].plot(np.arange(data.shape[1]), results['aic_densities'][1], label='Smoothed', color='orange')
    plt.legend()
    plt.savefig('../plots/test_gfd.pdf')
