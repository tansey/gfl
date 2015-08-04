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
import time
import sys
import argparse
import csv
from solver import TrailSolver
from utils import *


def main():
    parser = argparse.ArgumentParser(description='Runs the graph-fused lasso (GFL) solver on a given dataset with a given edge set. The GFL problem is defined as finding Beta such that it minimizes the equation f(Y, Beta) + lambda * g(E, Beta) where f is a smooth, convex loss function, typically 1/2 sum_i (y_i - Beta_i)^2, and g is sum of first differences on edges, sum_(s,t) |Beta_s - Beta_t| for each edge (s,t) in E.')

    parser.add_argument('data', help='The CSV file containing the vector of data points.')
    parser.add_argument('edges', help='The CSV file containing the edges connecting the variables, with one edge per line.')
    parser.add_argument('--output', '--o', help='The file to output the results.')
    
    parser.add_argument('--trails', help='A pre-computed set of trails. If not specified, the graph will be decomposed automatically.')
    parser.add_argument('--lam', type=float, help='If specified, uses a fixed value of the lambda penalty parameter. Otherwise, a solution path will be used to auto-tune lambda.')

    parser.add_argument('--minlam', type=float, default=0.2, help='The minimum lambda to try in the solution path grid.')
    parser.add_argument('--maxlam', type=float, default=1000.0, help='The maximum lambda to try in the solution path grid.')
    parser.add_argument('--numlam', type=int, default=30, help='The number of lambda values to try in the solution path grid.')
    parser.add_argument('--alpha', type=float, default=2., help='The ADMM alpha parameter.')
    parser.add_argument('--inflate', type=float, default=2., help='The ADMM varying alpha inflation rate.')
    parser.add_argument('--converge', type=float, default=1e-6, help='The convergence precision.')
    parser.add_argument('--maxsteps', type=int, default=1000000, help='The maximum number of ADMM steps before stopping.')

    parser.add_argument('--verbose', type=int, default=0, help='The level of print statements. 0=none, 1=moderate, 2=all. Default=0.')
    parser.add_argument('--time', action='store_true', help='Time the algorithm.')

    parser.set_defaults()

    args = parser.parse_args()

    ########### Load data from file
    if args.verbose:
        print 'Loading data'
    
    y = np.loadtxt(args.data, delimiter=',')
    edges = load_edges(args.edges)

    ########### Load trails from file if available, otherwise generate them automatically via the Eulerian pseudo-tour
    if args.trails:
        if args.verbose:
            print 'Loading trails from {0}'.format(args.trails)
        ntrails, trails, breakpoints, edges = load_trails(args.trails)
    else:
        if args.verbose:
            print 'Decomposing graph into trails'
        raise Exception('Not implemented yet.')

    if args.verbose:
        print 'Solving the GFL for {0} variables with {1} edges'.format(len(y), num_edges(edges))
    
    ########### Setup the solver
    solver = TrailSolver(args.alpha, args.inflate, args.maxsteps, args.converge)

    # Set the data and pre-cache any necessary structures
    solver.set_data(y, edges, ntrails, trails, breakpoints)

    ########### Run the C solver
    t0 = time.clock()
    if args.lam:
        beta = solver.solve(args.lam)
    else:
        beta = solver.solution_path(args.minlam, args.maxlam, args.numlam, verbose=max(0, args.verbose-1))['best']
    t1 = time.clock()

    ########### Print the timing stats
    if args.time:
        print 'Solved the GFL in {0}s and {1} total steps of ADMM.'.format(t1 - t0, np.array(solver.steps).sum())
    
    ########### Save the results to file
    if args.output:
        if args.verbose:
            print 'Saving results to {0}'.format(args.output)
        np.savetxt(args.output, beta, delimiter=',')





