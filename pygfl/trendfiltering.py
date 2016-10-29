'''Copyright (C) 2016 by Wesley Tansey

    This file is part of the GFL library.

    The GFL library is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    The GFL library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with the GFL library.  If not, see <http://www.gnu.org/licenses/>.
'''
import matplotlib.pylab as plt
import numpy as np
from numpy.ctypeslib import ndpointer
from scipy.sparse import coo_matrix, csr_matrix
from collections import defaultdict
from ctypes import *
from utils import *

'''Load the graph trend filtering library'''
graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
weighted_graphtf = graphfl_lib.graph_trend_filtering_weight_warm
weighted_graphtf.restype = c_int
weighted_graphtf.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'), c_double,
                    c_int, c_int, c_int,
                    ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                    c_int, c_double,
                    ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

weighted_graphtf_logit = graphfl_lib.graph_trend_filtering_logit_warm
weighted_graphtf_logit.restype = c_int
weighted_graphtf_logit.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), c_double,
                    c_int, c_int, c_int,
                    ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                    c_int, c_double,
                    ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

weighted_graphtf_poisson = graphfl_lib.graph_trend_filtering_poisson_warm
weighted_graphtf_poisson.restype = c_int
weighted_graphtf_poisson.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), c_double,
                    c_int, c_int, c_int,
                    ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                    c_int, c_double,
                    ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

class TrendFilteringSolver:
    def __init__(self, maxsteps=3000, converge=1e-6):
        self.maxsteps = maxsteps
        self.converge = converge

    def set_data(self, D, k, y, weights=None):
        self.y = y
        self.weights = weights if weights is not None else np.ones(len(self.y), dtype='double')
        self.initialize(D, k)

    def initialize(self, D, k):
        self.nnodes = len(self.y)
        self.D = D
        self.k = k
        self.Dk = get_delta(D, k).tocoo()
        self.Dk_minus_one = get_delta(self.D, self.k-1) if self.k > 0 else None
        self.beta = np.zeros(self.nnodes, dtype='double')
        self.steps = []
        self.u = np.zeros(self.Dk.shape[0], dtype='double')
        self.edges = None

    def solve(self, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        s = weighted_graphtf(self.nnodes, self.y, self.weights, lam,
                             self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                             self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                             self.maxsteps, self.converge,
                             self.beta, self.u)
        self.steps.append(s)
        return self.beta

    def solution_path(self, min_lambda, max_lambda, lambda_bins, verbose=0):
        '''Follows the solution path to find the best lambda value.'''
        self.u = np.zeros(self.Dk.shape[0], dtype='double')
        lambda_grid = np.exp(np.linspace(np.log(max_lambda), np.log(min_lambda), lambda_bins))
        aic_trace = np.zeros(lambda_grid.shape) # The AIC score for each lambda value
        aicc_trace = np.zeros(lambda_grid.shape) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros(lambda_grid.shape) # The BIC score for each lambda value
        dof_trace = np.zeros(lambda_grid.shape) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros(lambda_grid.shape)
        beta_trace = []
        best_idx = None
        best_plateaus = None

        if self.edges is None:
            self.edges = defaultdict(list)
            elist = csr_matrix(self.D).indices.reshape((self.D.shape[0], 2))
            for n1, n2 in elist:
                self.edges[n1].append(n2)
                self.edges[n2].append(n1)

        # Solve the series of lambda values with warm starts at each point
        for i, lam in enumerate(lambda_grid):
            if verbose:
                print '#{0} Lambda = {1}'.format(i, lam)

            # Fit to the final values
            beta = self.solve(lam)

            if verbose:
                print 'Calculating degrees of freedom'

            # Count the number of free parameters in the grid (dof) -- TODO: the graph trend filtering paper seems to imply we shouldn't multiply by (k+1)?
            dof_vals = self.Dk_minus_one.dot(beta) if self.k > 0 else beta
            plateaus = calc_plateaus(dof_vals, self.edges, rel_tol=0.01) if (self.k % 2) == 0 else nearly_unique(dof_vals, rel_tol=0.03)
            dof_trace[i] = max(1,len(plateaus)) #* (k+1)

            if verbose:
                print 'Calculating Information Criteria'

            # Get the negative log-likelihood
            log_likelihood_trace[i] = -0.5 * ((self.y - beta)**2).sum()

            # Calculate AIC = 2k - 2ln(L)
            aic_trace[i] = 2. * dof_trace[i] - 2. * log_likelihood_trace[i]
            
            # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
            aicc_trace[i] = aic_trace[i] + 2 * dof_trace[i] * (dof_trace[i]+1) / (len(beta) - dof_trace[i] - 1.)

            # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
            bic_trace[i] = -2 * log_likelihood_trace[i] + dof_trace[i] * (np.log(len(beta)) - np.log(2 * np.pi))

            # Track the best model thus far
            if best_idx is None or bic_trace[i] < bic_trace[best_idx]:
                best_idx = i
                best_plateaus = plateaus

            # Save the trace of all the resulting parameters
            beta_trace.append(np.array(beta))
            
            if verbose:
                print 'DoF: {0} AIC: {1} AICc: {2} BIC: {3}\n'.format(dof_trace[i], aic_trace[i], aicc_trace[i], bic_trace[i])

        if verbose:
            print 'Best setting (by BIC): lambda={0} [DoF: {1}, AIC: {2}, AICc: {3} BIC: {4}]'.format(lambda_grid[best_idx], dof_trace[best_idx], aic_trace[best_idx], aicc_trace[best_idx], bic_trace[best_idx])

        return {'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace,
                'dof': dof_trace,
                'loglikelihood': log_likelihood_trace,
                'beta': np.array(beta_trace),
                'lambda': lambda_grid,
                'best_idx': best_idx,
                'best': beta_trace[best_idx],
                'plateaus': best_plateaus}

class LogitTrendFilteringSolver(TrendFilteringSolver):
    def __init__(self, maxsteps=3000, converge=1e-6):
        self.maxsteps = maxsteps
        self.converge = converge

    def set_data(self, D, k, trials, successes):
        self.trials = trials
        self.successes = successes
        self.initialize(D, k)

    def solve(self, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        s = weighted_graphtf_logit(self.nnodes, self.trials, self.successes, lam,
                                 self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                                 self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                                 self.maxsteps, self.converge,
                                 self.beta, self.u)
        self.steps.append(s)
        return self.beta

class PoissonTrendFilteringSolver(TrendFilteringSolver):
    def __init__(self, maxsteps=3000, converge=1e-6):
        self.maxsteps = maxsteps
        self.converge = converge

    def set_data(self, D, k, obs):
        self.obs = obs
        self.initialize(D, k)

    def solve(self, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        s = weighted_graphtf_poisson(self.nnodes, self.obs, lam,
                                 self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                                 self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                                 self.maxsteps, self.converge,
                                 self.beta, self.u)
        self.steps.append(s)
        return self.beta


def test_solve_gtf():
    # Load the data and create the penalty matrix
    max_k = 3
    y = (np.sin(np.linspace(-np.pi, np.pi, 100)) + 1) * 5
    y[25:75] += np.sin(np.linspace(1.5*-np.pi, np.pi*2, 50))*5 ** (np.abs(np.arange(50) / 25.))
    y += np.random.normal(0,1.0,size=len(y))
    # np.savetxt('/Users/wesley/temp/tfdata.csv', y, delimiter=',')
    # y = np.loadtxt('/Users/wesley/temp/tfdata.csv', delimiter=',')
    mean_offset = y.mean()
    y -= mean_offset
    stdev_offset = y.std()
    y /= stdev_offset
    
    # equally weight each data point
    w = np.ones(len(y))

    lam = 50.

    # try different weights for each data point
    # w = np.ones(len(y))
    # w[0:len(y)/2] = 1.
    # w[len(y)/2:] = 100.
    
    D = coo_matrix(get_1d_penalty_matrix(len(y)))

    z = np.zeros((max_k,len(y)))
    tf = TrendFilteringSolver()
    tf.set_data(y, D, w)
    for k in xrange(max_k):
        #z[k] = tf.solve(k, lam)
        z[k] = tf.solution_path(k, 0.2, 2000, 100, verbose=True)['best']
    
    y *= stdev_offset
    y += mean_offset
    z *= stdev_offset
    z += mean_offset


    colors = ['orange', 'skyblue', '#009E73', 'purple']
    fig, ax = plt.subplots(max_k)
    x = np.linspace(0,1,len(y))
    for k in xrange(max_k):
        ax[k].scatter(x, y, alpha=0.5)
        ax[k].plot(x, z[k], lw=2, color=colors[k], label='k={0}'.format(k))
        ax[k].set_xlim([0,1])
        ax[k].set_ylabel('y')
        ax[k].set_title('k={0}'.format(k))
    
    plt.show()
    plt.clf()

def test_solve_gtf_logit():
    max_k = 5
    trials = np.random.randint(5, 30, size=100).astype('int32')
    probs = np.zeros(100)
    probs[:25] = 0.25
    probs[25:50] = 0.75
    probs[50:75] = 0.5
    probs[75:] = 0.1
    successes = np.array([np.random.binomial(t, p) for t,p in zip(trials, probs)]).astype('int32')

    lam = 3.

    D = coo_matrix(get_1d_penalty_matrix(len(trials)))
    z = np.zeros((max_k,len(trials)))
    for k in xrange(max_k):
        tf = LogitTrendFilteringSolver()
        tf.set_data(trials, successes, D)
        z[k] = tf.solve(k, lam)

    colors = ['orange', 'skyblue', '#009E73', 'purple', 'black']
    fig, ax = plt.subplots(max_k+1)
    x = np.linspace(0,1,len(trials))
    ax[0].bar(x, successes, width=1./len(x), color='darkblue', alpha=0.3)
    ax[0].bar(x, trials-successes, width=1./len(x), color='skyblue', alpha=0.3, bottom=successes)
    ax[0].set_ylim([0,30])
    ax[0].set_xlim([0,1])
    ax[0].set_ylabel('Trials and successes')
    for k in xrange(max_k):
        ax[k+1].scatter(x, probs, alpha=0.5)
        ax[k+1].plot(x, z[k], lw=2, color=colors[k], label='k={0}'.format(k))
        ax[k+1].set_ylim([0,1])
        ax[k+1].set_xlim([0,1])
        ax[k+1].set_ylabel('Probability of success')
    
    plt.show()
    plt.clf()

def test_solve_gtf_poisson():
    max_k = 5
    probs = np.zeros(100)
    probs[:25] = 5.
    probs[25:50] = 9.
    probs[50:75] = 3.
    probs[75:] = 6.
    obs = np.array([np.random.poisson(p) for p in probs]).astype('int32')

    lam = 5.

    D = coo_matrix(get_1d_penalty_matrix(len(obs)))
    z = np.zeros((max_k,len(obs)))
    for k in xrange(max_k):
        tf = PoissonTrendFilteringSolver()
        tf.set_data(obs, D)
        z[k] = tf.solve(k, lam)

    colors = ['orange', 'skyblue', '#009E73', 'purple', 'black']
    fig, ax = plt.subplots(max_k+1)
    x = np.linspace(0,1,len(obs))
    ax[0].bar(x, obs, width=1./len(x), color='darkblue', alpha=0.3)
    ax[0].set_xlim([0,1])
    ax[0].set_ylabel('Observations')
    for k in xrange(max_k):
        ax[k+1].scatter(x, probs, alpha=0.5)
        ax[k+1].plot(x, z[k], lw=2, color=colors[k], label='k={0}'.format(k))
        ax[k+1].set_xlim([0,1])
        ax[k+1].set_ylabel('Beta (k={0})'.format(k))
    
    plt.show()
    plt.clf()

if __name__ == '__main__':
    test_solve_gtf()
