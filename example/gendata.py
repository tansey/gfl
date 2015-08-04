import numpy as np
from utils import *


def num_nodes(edges):
    s = set()
    for k,v in edges.iteritems():
        s |= set(v)
    return np.max(list(s))+1

plateau_vals = [30, 45, 10, 20]
edges = load_edges('../example/edges.csv')
nnodes = num_nodes(edges)
data = np.zeros(nnodes)
plateau_size = max(2, nnodes / 20)

# Create random plateaus
create_plateaus(data, edges, plateau_size, plateau_vals)

# Noise the data
y = np.zeros(data.shape, dtype='double')
y[:] = data + np.random.normal(0, 3, size=y.shape)

# Center and standardize the data
data = (data - data.mean()) / data.std()
y = (y - y.mean()) / y.std()

np.savetxt('../example/truth.csv', data, delimiter=',')
np.savetxt('../example/data.csv', y, delimiter=',')
