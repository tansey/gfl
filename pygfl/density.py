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
from scipy.stats import binom
from scipy.interpolate import interp1d
from collections import deque
from ctypes import *
from bayes import sample_gtf
from utils import *

class GraphFusedDensity:
    def __init__(self, dof_tolerance=1e-4, converge=1e-6, max_steps=100,
                       alpha=2., inflate=2., polya_levels=7, 
                       min_lambda=0.20, max_lambda=1.5, lambda_bins=30,
                       interpolate=False,
                       verbose=0, bins_allowed=None):
        self.dof_tolerance = dof_tolerance
        self.converge = converge
        self.max_steps = max_steps
        self.alpha = alpha
        self.inflate = inflate
        self.polya_levels = polya_levels
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.lambda_bins = lambda_bins
        self.interpolate = interpolate
        self.verbose = verbose
        self.bins_allowed = bins_allowed

        # Load the graph fused lasso library
        self.graphfl_lib = cdll.LoadLibrary('libgraphfl.so')
        self.graphfl = self.graphfl_lib.graph_fused_lasso_logit_warm
        self.graphfl.restype = c_int
        self.graphfl.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                                        c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                                        c_double, c_double, c_double,
                                        c_int, c_double,
                                        ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                                        ndpointer(c_double, flags='C_CONTIGUOUS')]
        self.graphtf = self.graphfl_lib.graph_trend_filtering_logit_warm
        self.graphtf.restype = c_int
        self.graphtf.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), c_double,
                                        c_int, c_int, c_int,
                                        ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                                        c_int, c_double,
                                        ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]


    def set_data(self, data, edges, k=0, ntrails=None, trails=None, breakpoints=None):
        self.data = data
        self.edges = edges
        self.k = k
        self.ntrails = ntrails
        self.trails = trails
        self.breakpoints = breakpoints
        self.map_density = None
        self.bayes_density = None
        self.map_betas = None
        self.bayes_betas = None
        self.num_nodes = data.shape[0]
        self.max_x = data.shape[1]

        self.D = matrix_from_edges(self.edges)
        self.Dk = get_delta(self.D, self.k).tocoo()
        self.Dk_minus_one = get_delta(self.D, self.k-1) if self.k > 0 else None

        # Create the Polya tree
        self.bins = []
        self.splits = set()
        self.polya_tree_descend(0, self.max_x, 0)
        if self.verbose:
            print 'bins: {0}'.format(len(self.bins))

    def polya_tree_descend(self, left, right, level):
        if (right - left) < 2:
            return

        if (left, right) in self.splits:
            return

        self.splits.add((left, right))
            
        # Just take the midpoint as the split
        mid = int((left + right) / 2)

        trials = self.data[:, np.arange(left, right)].sum(axis=1, dtype='int32')
        successes = self.data[:, np.arange(left, mid)].sum(axis=1, dtype='int32')
        
        self.bins.append((left, mid, right, trials, successes))

        if level < self.polya_levels and trials.sum() > 0:
            # Recurse left
            self.polya_tree_descend(left, mid, level+1)

            # Recurse right
            self.polya_tree_descend(mid, right, level+1)

    def solution_path(self):
        '''Follows the solution path of the generalized lasso to find the best lambda value.'''
        lambda_grid = np.exp(np.linspace(np.log(self.max_lambda), np.log(self.min_lambda), self.lambda_bins))
        aic_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The AIC score for each lambda value
        aicc_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The AICc score for each lambda value (correcting for finite sample size)
        bic_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The BIC score for each lambda value
        dof_trace = np.zeros((len(self.bins),lambda_grid.shape[0])) # The degrees of freedom of each final solution
        log_likelihood_trace = np.zeros((len(self.bins),lambda_grid.shape[0]))
        bic_best_idx = [None for _ in self.bins]
        aic_best_idx = [None for _ in self.bins]
        aicc_best_idx = [None for _ in self.bins]
        bic_best_betas = [None for _ in self.bins]
        aic_best_betas = [None for _ in self.bins]
        aicc_best_betas = [None for _ in self.bins]
        if self.k == 0 and self.trails is not None:
            betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
            zs = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
            us = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
        else:
            betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
            us = [np.zeros(self.Dk.shape[0], dtype='double') for _ in self.bins]
        for i, _lambda in enumerate(lambda_grid):
            if self.verbose:
                print '\n#{0} Lambda = {1}'.format(i, _lambda)

            # Run the graph fused lasso over each bin with the current lambda value
            initial_values = (betas, zs, us) if self.k == 0 and self.trails is not None else (betas, us)
            self.run(_lambda, initial_values=initial_values)

            if self.verbose > 1:
                print '\tCalculating degrees of freedom and information criteria'

            for b, beta in enumerate(betas):
                if self.bins_allowed is not None and b not in self.bins_allowed:
                    continue

                # Count the number of free parameters in the grid (dof)
                # TODO: this is not really the true DoF, since a change in a higher node multiplies
                # the DoF in the lower nodes
                # dof_trace[b,i] = len(self.calc_plateaus(beta))
                dof_vals = self.Dk_minus_one.dot(beta) if self.k > 0 else beta
                plateaus = calc_plateaus(dof_vals, self.edges, rel_tol=0.01) if (self.k % 2) == 0 else nearly_unique(dof_vals, rel_tol=0.03)
                #plateaus = calc_plateaus(dof_vals, self.edges, rel_tol=1e-5) if (self.k % 2) == 0 else nearly_unique(dof_vals, rel_tol=1e-5)
                dof_trace[b,i] = max(1,len(plateaus)) #* (k+1)

                # Get the negative log-likelihood
                log_likelihood_trace[b,i] = self.data_log_likelihood(self.bins[b][-1], self.bins[b][-2], beta)

                # Calculate AIC = 2k - 2ln(L)
                aic_trace[b,i] = 2. * dof_trace[b,i] - 2. * log_likelihood_trace[b,i]
                
                # Calculate AICc = AIC + 2k * (k+1) / (n - k - 1)
                aicc_trace[b,i] = aic_trace[b,i] + 2 * dof_trace[b,i] * (dof_trace[b,i]+1) / (self.num_nodes - dof_trace[b,i] - 1.)

                # Calculate BIC = -2ln(L) + k * (ln(n) - ln(2pi))
                bic_trace[b,i] = -2 * log_likelihood_trace[b,i] + dof_trace[b,i] * (np.log(self.num_nodes) - np.log(2 * np.pi))

                # Track the best model thus far
                if aic_best_idx[b] is None or aic_trace[b,i] < aic_trace[b,aic_best_idx[b]]:
                    aic_best_idx[b] = i
                    aic_best_betas[b] = np.array(beta)

                # Track the best model thus far
                if aicc_best_idx[b] is None or aicc_trace[b,i] < aicc_trace[b,aicc_best_idx[b]]:
                    aicc_best_idx[b] = i
                    aicc_best_betas[b] = np.array(beta)

                # Track the best model thus far
                if bic_best_idx[b] is None or bic_trace[b,i] < bic_trace[b,bic_best_idx[b]]:
                    bic_best_idx[b] = i
                    bic_best_betas[b] = np.array(beta)

                if self.verbose and self.bins_allowed is not None:
                    print '\tBin {0} Log-Likelihood: {1} DoF: {2} AIC: {3} AICc: {4} BIC: {5}'.format(b, log_likelihood_trace[b,i], dof_trace[b,i], aic_trace[b,i], aicc_trace[b,i], bic_trace[b,i])

            if self.verbose and self.bins_allowed is None:
                print 'Overall Log-Likelihood: {0} DoF: {1} AIC: {2} AICc: {3} BIC: {4}'.format(log_likelihood_trace[:,i].sum(), dof_trace[:,i].sum(), aic_trace[:,i].sum(), aicc_trace[:,i].sum(), bic_trace[:,i].sum())

        if self.verbose:
            print ''
            print 'Best settings per bin:'
            for b, (aic_idx, aicc_idx, bic_idx) in enumerate(zip(aic_best_idx, aicc_best_idx, bic_best_idx)):
                if self.bins_allowed is not None and b not in self.bins_allowed:
                    continue
                left, mid, right, trials, successes = self.bins[b]
                print '\tBin #{0} ([{1}, {2}], split={3}) lambda: AIC={4:.2f} AICC={5:.2f} BIC={6:.2f} DoF: AIC={7:.0f} AICC={8:.0f} BIC={9:.0f}'.format(
                        b, left, right, mid,
                        lambda_grid[aic_idx], lambda_grid[aicc_idx], lambda_grid[bic_idx],
                        dof_trace[b,aic_idx], dof_trace[b,aicc_idx], dof_trace[b,bic_idx])
            print ''

        if self.bins_allowed is None:
            if self.verbose:
                print 'Creating densities from betas...'
            bic_density = self.density_from_betas(bic_best_betas)
            aic_density = self.density_from_betas(aic_best_betas)
            aicc_density = self.density_from_betas(aicc_best_betas)
            self.map_density = bic_density
        else:
            aic_density, aicc_density, bic_density = None, None, None
        
        self.map_betas = bic_best_betas

        return {'aic': aic_trace,
                'aicc': aicc_trace,
                'bic': bic_trace,
                'dof': dof_trace,
                'loglikelihood': log_likelihood_trace,
                'lambdas': lambda_grid,
                'aic_betas': aic_best_betas,
                'aicc_betas': aicc_best_betas,
                'bic_betas': bic_best_betas,
                'aic_best_idx': aic_best_idx,
                'aicc_best_idx': aicc_best_idx,
                'bic_best_idx': bic_best_idx,
                'aic_densities': aic_density,
                'aicc_densities': aicc_density,
                'bic_densities': bic_density}

    def estimate_change_points(self):
        if self.map_betas is None:
            self.solution_path()
        return np.array([(np.array(self.Dk.dot(betas)) > 0.04).sum() for betas in self.map_betas])

    def bayes_estimate(self, prior='laplacegamma',
                        empirical=False, lam0=None, explore_iterations=1000, explore_burn=100, explore_thin=2,
                        iterations=10000, burn=2000, thin=5):
        # if lam0 is None and empirical:
        #     change_points = self.estimate_change_points()
        #     lam0 = ((change_points+1.0) / float(self.D.shape[0]))**(float(self.D.shape[1])/float(self.D.shape[0]))
        
        sample_size = (iterations - burn) / thin
        
        if self.bins_allowed is None:
            beta_samples = np.array([np.zeros((sample_size, self.D.shape[1]), dtype='double') for _ in self.bins])
            lam_samples = np.array([np.zeros(sample_size, dtype='double') for _ in self.bins])
            target_bins = self.bins
        else:
            beta_samples = np.array([np.zeros((sample_size, self.D.shape[1]), dtype='double') for _ in self.bins_allowed])
            lam_samples = np.array([np.zeros(sample_size, dtype='double') for _ in self.bins_allowed])
            target_bins = [self.bins[x] for x in self.bins_allowed]

        for j, (left, mid, right, trials, successes) in enumerate(target_bins):
            if self.verbose:
                print 'Bin #{0}'.format(j if self.bins_allowed is None else self.bins_allowed[j])

            if empirical:
                best_dic = None
                best_bic = None
                best_lam = None
                for lam in lam0[j]:
                    betas, lams = sample_gtf((trials, successes), self.D, self.k,
                                           likelihood='binomial', prior=prior,
                                           empirical=empirical, lam0=lam,
                                           iterations=explore_iterations, burn=explore_burn, thin=explore_thin,
                                           verbose=self.verbose)
                    mean_beta = betas.mean(axis=0)
                    log_likelihood = self.data_log_likelihood(successes, trials, mean_beta)
                    expected_deviance = -2 * np.mean([self.data_log_likelihood(successes, trials, b) for b in betas])
                    dof = expected_deviance + 2 * log_likelihood
                    dic = expected_deviance + dof
                    likelihood_var = np.var([self.data_log_likelihood(successes, trials, b) for b in betas])
                    dof_vals = self.Dk_minus_one.dot(mean_beta) if self.k > 0 else mean_beta
                    plateaus = calc_plateaus(dof_vals, self.edges, rel_tol=0.01) if (self.k % 2) == 0 else nearly_unique(dof_vals, rel_tol=0.03)
                    bic_dof = max(1,len(plateaus))
                    bic = -2 * log_likelihood + bic_dof * (np.log(self.num_nodes) - np.log(2 * np.pi))
                    if self.verbose > 2:
                        print '\tlambda: {0} E[D]: {1} DoF: {2} DIC: {3} LikelihoodVariance: {4} BIC-DoF: {5} BIC: {6}'.format(lam, expected_deviance, dof, dic, likelihood_var, bic_dof, bic)
                    if best_dic is None or dic < best_dic:
                        best_dic = dic
                    if best_bic is None or bic < best_bic:
                        best_bic = bic
                        best_lam = lam
                if self.verbose:
                    print 'Empirical Bayes lambda choice: {0} (DIC: {1})'.format(best_lam, best_dic)
                lam0 = best_lam
            
            beta, lam = sample_gtf((trials, successes), self.D, self.k,
                                   likelihood='binomial', prior=prior,
                                   empirical=empirical, lam0=lam0,
                                   iterations=iterations, burn=burn, thin=thin,
                                   verbose=self.verbose)

            beta_samples[j] = -np.log(1./np.clip(beta, 1e-12, 1-1e-12) - 1.) # convert back to natural parameter form
            lam_samples[j] = lam

        if self.bins_allowed is None:
            if self.verbose:
                print 'Creating densities from betas...'
            means = beta_samples.mean(axis=1)
            self.bayes_density = self.density_from_betas(means)
            self.bayes_betas = means

        return {'betas': beta_samples, 'lambdas': lam_samples, 'density': self.bayes_density}

    def density_from_betas(self, betas):
        if self.interpolate:
            # Calculate lowest level bins
            x = np.array(sorted(set([left for left, mid, right, trials, successes in self.bins] + [mid for left, mid, right, trials, successes in self.bins] + [self.max_x-1])), dtype="double")
            y = np.ones((self.num_nodes, len(x)))
            for (left, mid, right, trials, successes), beta in zip(self.bins, betas):
                p = 1. / (1+np.exp(-beta))
                y[:,np.logical_and(x >= left, x < mid)] *= p[:,np.newaxis]
                y[:,np.logical_and(x >= mid, x < right)] *= 1-p[:,np.newaxis]
            y /= y.sum(axis=1)[:,np.newaxis]
            
            # Interpolate points to form the entire density
            d = interp1d(x, y)(np.arange(self.max_x))
            d /= d.sum(axis=1)[:,np.newaxis] # re-normalize

            return d
        else:
            y = np.ones((self.num_nodes, self.max_x))
            for (left, mid, right, trials, successes), beta in zip(self.bins, betas):
                p = 1. / (1+np.exp(-beta))
                y[:,left:mid] *= p[:,np.newaxis]
                y[:,mid:right] *= 1-p[:,np.newaxis]
            return y / y.sum(axis=1)[:,np.newaxis]


    def run(self, lam, initial_values=None):
        '''Run the graph-fused logit lasso with a fixed lambda penalty.'''
        if initial_values is not None:
            if self.k == 0 and self.trails is not None:
                betas, zs, us = initial_values
            else:
                betas, us = initial_values
        else:
            if self.k == 0 and self.trails is not None:
                betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
                zs = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
                us = [np.zeros(self.breakpoints[-1], dtype='double') for _ in self.bins]
            else:
                betas = [np.zeros(self.num_nodes, dtype='double') for _ in self.bins]
                us = [np.zeros(self.Dk.shape[0], dtype='double') for _ in self.bins]

        for j, (left, mid, right, trials, successes) in enumerate(self.bins):
            if self.bins_allowed is not None and j not in self.bins_allowed:
                continue

            if self.verbose > 2:
                print '\tBin #{0} [{1},{2},{3}]'.format(j, left, mid, right)
            # if self.verbose > 3:
            #     print 'Trials:\n{0}'.format(pretty_str(trials))
            #     print ''
            #     print 'Successes:\n{0}'.format(pretty_str(successes))
                
            beta = betas[j]
            u = us[j]

            if self.k == 0 and self.trails is not None:
                z = zs[j]
                # Run the graph-fused lasso algorithm
                self.graphfl(len(beta), trials, successes,
                             self.ntrails, self.trails, self.breakpoints,
                             lam, self.alpha, self.inflate,
                             self.max_steps, self.converge,
                             beta, z, u)
            else:
                # Run the graph trend filtering algorithm
                self.graphtf(len(beta), trials, successes, lam,
                                 self.Dk.shape[0], self.Dk.shape[1], self.Dk.nnz,
                                 self.Dk.row.astype('int32'), self.Dk.col.astype('int32'), self.Dk.data.astype('double'),
                                 self.max_steps, self.converge,
                                 beta, u)
                beta = np.clip(beta, 1e-12, 1-1e-12) # numerical stability
                betas[j] = -np.log(1./beta - 1.) # convert back to natural parameter form

        return (betas, zs, us) if self.k == 0 and self.trails is not None else (betas, us)

    def data_log_likelihood(self, successes, trials, beta):
        '''Calculates the log-likelihood of a Polya tree bin given the beta values.'''
        return binom.logpmf(successes, trials, 1.0 / (1 + np.exp(-beta))).sum()

    def calc_plateaus(self, beta):
        '''Calculate the plateaus (degrees of freedom) a graph of beta values in linear time.'''
        to_check = deque(xrange(len(beta)))
        check_map = np.zeros(beta.shape, dtype=bool)
        check_map[np.isnan(beta)] = True
        plateaus = []

        if self.verbose > 2:
            print '\tCalculating plateaus...'

        if self.verbose > 2:
            print '\tIndices to check {0} {1}'.format(len(to_check), check_map.shape)

        # Loop until every beta index has been checked
        while to_check:
            if self.verbose > 2:
                print '\t\tPlateau #{0}'.format(len(plateaus) + 1)

            # Get the next unchecked point on the grid
            idx = to_check.popleft()

            # If we already have checked this one, just pop it off
            while to_check and check_map[idx]:
                try:
                    idx = to_check.popleft()
                except:
                    break

            # Edge case -- If we went through all the indices without reaching an unchecked one.
            if check_map[idx]:
                break

            # Create the plateau and calculate the inclusion conditions
            cur_plateau = set([idx])
            cur_unchecked = deque([idx])
            val = beta[idx]
            min_member = val - self.dof_tolerance
            max_member = val + self.dof_tolerance

            # Check every possible boundary of the plateau
            while cur_unchecked:
                idx = cur_unchecked.popleft()
                
                # neighbors to check
                local_check = []

                # Generic graph case
                local_check.extend(self.edges[idx])

                # Check the index's unchecked neighbors
                for local_idx in local_check:
                    if not check_map[local_idx] \
                        and beta[local_idx] >= min_member \
                        and beta[local_idx] <= max_member:
                            # Label this index as being checked so it's not re-checked unnecessarily
                            check_map[local_idx] = True

                            # Add it to the plateau and the list of local unchecked locations
                            cur_unchecked.append(local_idx)
                            cur_plateau.add(local_idx)

            # Track each plateau's indices
            plateaus.append((val, cur_plateau))

        # Returns the list of plateaus and their values
        return plateaus