'''Copyright (C) 2015 by Wesley Tansey

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
import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import *
from utils import *

''' Load the graph fused lasso library '''
graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
graphfl = graphfl_lib.graph_fused_lasso_warm
graphfl.restype = c_int
graphfl.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'),
                    c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                    c_double, c_double, c_double, c_int, c_double,
                    ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

class TrailSolver:
    def __init__(self, alpha=2., inflate=2., maxsteps=1000000, converge=1e-6):
        self.alpha = alpha
        self.inflate = inflate
        self.maxsteps = maxsteps
        self.converge = converge

    def set_data(self, y, edges, ntrails, trails, breakpoints):
        self.y = y
        self.edges = edges
        self.nnodes = len(y)
        self.ntrails = ntrails
        self.trails = trails
        self.breakpoints = breakpoints
        self.beta = np.zeros(self.nnodes, dtype='double')
        self.z = np.zeros(self.breakpoints[-1], dtype='double')
        self.u = np.zeros(self.breakpoints[-1], dtype='double')
        self.steps = []

    def solve(self, lam):
        '''Solves the GFL for a fixed value of lambda.'''
        s = graphfl(self.nnodes, self.y,
                        self.ntrails, self.trails, self.breakpoints,
                        lam,
                        self.alpha, self.inflate, self.maxsteps, self.converge,
                        self.beta, self.z, self.u)
        self.steps.append(s)
        return self.beta

    def solution_path(self, min_lambda, max_lambda, lambda_bins, verbose=0):
        '''Follows the solution path to find the best lambda value.'''
        lambda_grid = np.exp(np.linspace(np.log(max_lambda), np.log(min_lambda), lambda_bins))
        aic_trace = np.zeros(lambda_grid.shape) # The AIC score for each lambda value
        aicc_trace = np.zeros(lambda_grid.shape) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros(lambda_grid.shape) # The BIC score for each lambda value
        dof_trace = np.zeros(lambda_grid.shape) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros(lambda_grid.shape)
        beta_trace = []
        best_idx = None
        best_plateaus = None

        # Solve the series of lambda values with warm starts at each point
        for i, lam in enumerate(lambda_grid):
            if verbose:
                print '#{0} Lambda = {1}'.format(i, lam)

            # Fit to the final values
            beta = self.solve(lam)

            if verbose:
                print 'Calculating degrees of freedom'

            # Count the number of free parameters in the grid (dof)
            plateaus = calc_plateaus(beta, self.edges)
            dof_trace[i] = len(plateaus)

            if verbose:
                print 'Calculating AIC'

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
                print 'DoF: {0} AIC: {1} AICc: {2} BIC: {3}'.format(dof_trace[i], aic_trace[i], aicc_trace[i], bic_trace[i])

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