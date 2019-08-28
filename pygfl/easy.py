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
from networkx import Graph
from pygfl.trails import decompose_graph
from pygfl.solver import TrailSolver
from pygfl.logistic_solver import LogisticTrailSolver
from pygfl.binomial_solver import BinomialTrailSolver
from pygfl.utils import *

import numpy as np

def solve_gfl(data, edges=None, weights=None,
              minlam=0.2, maxlam=1000.0, numlam=30,
              alpha=0.2, inflate=2., converge=1e-6,
              maxsteps=100000, lam=None, verbose=0,
              missing_val=None, full_path=False,
              loss='normal'):
    '''A very easy-to-use version of GFL solver that just requires the data and
    the edges.'''

    #Fix no edge cases
    if edges is not None and edges.shape[0] == 0:
        return data

    if verbose:
        print('Decomposing graph into trails')

    if loss == 'binomial':
        flat_data = data[0].flatten()
        nonmissing_flat_data = flat_data, data[1].flatten()
    else:
        flat_data = data.flatten()
        nonmissing_flat_data = flat_data

    if weights is not None:
        weights = weights.flatten()

    if edges is None:
        if loss == 'binomial':
            if verbose:
                print('Using default edge set of a grid of same shape as the data: {0}'.format(data[0].shape))
            edges = hypercube_edges(data[0].shape)
        else:
            if verbose:
                print('Using default edge set of a grid of same shape as the data: {0}'.format(data.shape))
            edges = hypercube_edges(data.shape)
        if missing_val is not None:
            if verbose:
                print('Removing all data points whose data value is {0}'.format(missing_val))
            edges = [(e1,e2) for (e1,e2) in edges if flat_data[e1] != missing_val and flat_data[e2] != missing_val]
            if loss == 'binomial':
                nonmissing_flat_data = flat_data[flat_data != missing_val], nonmissing_flat_data[1][flat_data != missing_val]
            else:
                nonmissing_flat_data = flat_data[flat_data != missing_val]

    # Keep initial edges
    init_edges = np.array(edges)

    ########### Setup the graph
    g = Graph()
    g.add_edges_from(edges)
    chains = decompose_graph(g, heuristic='greedy')
    ntrails, trails, breakpoints, edges = chains_to_trails(chains)

    if verbose:
        print('Setting up trail solver')

    ########### Setup the solver
    if loss == 'normal':
        solver = TrailSolver(alpha, inflate, maxsteps, converge)
    elif loss == 'logistic':
        solver = LogisticTrailSolver(alpha, inflate, maxsteps, converge)
    elif loss == 'binomial':
        solver = BinomialTrailSolver(alpha, inflate, maxsteps, converge)
    else:
        raise NotImplementedError('Loss must be normal, logistic, or binomial')

    # Set the data and pre-cache any necessary structures
    solver.set_data(nonmissing_flat_data, edges, ntrails, trails, breakpoints, weights=weights)

    if verbose:
        print('Solving')

    ########### Run the solver
    if lam:
        # Fixed lambda
        beta = solver.solve(lam)
    else:
        # Grid search to find the best lambda
        beta = solver.solution_path(minlam, maxlam, numlam, verbose=max(0, verbose-1))
        if not full_path:
            beta = beta['best']

    ########### Fix disconnected nodes
    mask = np.ones_like(beta)
    mask[init_edges[:,0]] = 0
    mask[init_edges[:,1]] = 0
    beta[mask>0] = data[mask>0]

    return beta

