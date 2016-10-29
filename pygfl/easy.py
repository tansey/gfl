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
from trails import decompose_graph
from solver import TrailSolver
from utils import *

def solve_gfl(data, edges, weights=None,
              minlam=0.2, maxlam=1000.0, numlam=30,
              alpha=0.2, inflate=2., converge=1e-6,
              maxsteps=1000000, lam=None, verbose=0):
    '''A very easy-to-use version of GFL solver that just requires the data and
    the edges.'''
    if verbose:
        print 'Decomposing graph into trails'

    ########### Setup the graph
    g = Graph()
    g.add_edges_from(edges)
    chains = decompose_graph(g, heuristic='greedy')
    ntrails, trails, breakpoints, edges = chains_to_trails(chains)

    if verbose:
        print 'Setting up trail solver'

    ########### Setup the solver
    solver = TrailSolver(alpha, inflate, maxsteps, converge)

    # Set the data and pre-cache any necessary structures
    solver.set_data(data, edges, ntrails, trails, breakpoints, weights=weights)

    if verbose:
        print 'Solving'

    ########### Run the solver
    if lam:
        # Fixed lambda
        beta = solver.solve(lam)
    else:
        # Grid search to find the best lambda
        beta = solver.solution_path(minlam, maxlam, numlam, verbose=max(0, verbose-1))['best']
    
    return beta

