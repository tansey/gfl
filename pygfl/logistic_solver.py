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

class LogisticTrailSolver(TrailSolver):
    def __init__(self, alpha=2., inflate=2., maxsteps=100000, converge=1e-6, penalty='gfl', max_dp_steps=5000, gamma=1.):
        TrailSolver.__init__(self, alpha, inflate, maxsteps, converge, penalty, max_dp_steps, gamma)

        if penalty != 'gfl':
            raise NotImplementedError('Only regular fused lasso supported for logistic loss.')

    def solve_gfl(self, lam):
        if hasattr(lam, '__len__'):
            raise NotImplementedError('Only uniform edge weighting implemented for logistic loss.')

        # Run the graph-fused lasso algorithm
        s = logistic_graphfl(self.nnodes, np.ones(self.nnodes, dtype='int32'), self.y.astype('int32'),
                     self.ntrails, self.trails, self.breakpoints,
                     lam, self.alpha, self.inflate,
                     self.maxsteps, self.converge,
                     self.beta, self.z, self.u)
        self.steps.append(s)
        return self.beta

    def log_likelihood(self, beta):
        signs = -(self.y * 2 - 1)
        return -np.log(1 + np.exp(signs * beta)).sum()





