import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import *
from pygfl.utils import *
from pygfl.solver import TrailSolver

# Load the graph fused lasso library
logistic_graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
logistic_graphfl = logistic_graphfl_lib.graph_fused_lasso_logit_warm
logistic_graphfl.restype = c_int
logistic_graphfl.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                                c_double, c_double, c_double,
                                c_int, c_double,
                                ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                                ndpointer(c_double, flags='C_CONTIGUOUS')]

class BinomialTrailSolver(TrailSolver):
    def __init__(self, alpha=2., inflate=2., maxsteps=100000, converge=1e-6, penalty='gfl', max_dp_steps=5000, gamma=1.):
        TrailSolver.__init__(self, alpha, inflate, maxsteps, converge, penalty, max_dp_steps, gamma)

        if penalty != 'gfl':
            raise NotImplementedError('Only regular fused lasso supported for logistic loss.')

    def set_data(self, y, edges, ntrails, trails, breakpoints, weights=None):
        self.y = y
        self.edges = edges if type(edges) is defaultdict else edge_map_from_edge_list(edges)
        self.nnodes = len(y[0])
        self.ntrails = ntrails
        self.trails = trails
        self.breakpoints = breakpoints
        self.weights = weights
        self.beta = np.zeros(self.nnodes, dtype='double')
        self.z = np.zeros(self.breakpoints[-1], dtype='double')
        self.u = np.zeros(self.breakpoints[-1], dtype='double')
        self.steps = []

    def solve_gfl(self, lam):
        if hasattr(lam, '__len__'):
            raise NotImplementedError('Only uniform edge weighting implemented for logistic loss.')

        trials = self.y[0].astype('int32')
        successes = self.y[1].astype('int32')

        # Run the graph-fused lasso algorithm
        s = logistic_graphfl(self.nnodes,
                     trials,
                     successes,
                     self.ntrails, self.trails, self.breakpoints,
                     lam, self.alpha, self.inflate,
                     self.maxsteps, self.converge,
                     self.beta, self.z, self.u)
        self.steps.append(s)
        return self.beta

    def log_likelihood(self, beta):
        trials = self.y[0].astype('int32')
        successes = self.y[1].astype('int32')
        return (successes * np.log(1. / (1. + np.exp(-beta))) + \
                (trials - successes) * np.log(1. / (1. + np.exp(beta)))).sum()





