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
from easy import solve_gfl

def main():
    parser = argparse.ArgumentParser(description='Runs the graph-fused lasso (GFL) solver on a given dataset with a given edge set. The GFL problem is defined as finding Beta such that it minimizes the equation f(Y, Beta) + lambda * g(E, Beta) where f is a smooth, convex loss function, typically 1/2 sum_i (y_i - Beta_i)^2, and g is sum of first differences on edges, sum_(s,t) |Beta_s - Beta_t| for each edge (s,t) in E.')

    parser.add_argument('data', help='The CSV file containing the vector of data points.')
    parser.add_argument('edges', help='The CSV file containing the edges connecting the variables, with one edge per line.')
    parser.add_argument('--output', '--o', help='The file to output the results.')

    parser.add_argument('--verbose', type=int, default=1, help='The level of print statements. 0=none, 1=moderate, 2=all. Default=0.')
    parser.add_argument('--time', action='store_true', help='Print the timing stats for the algorithm.')

    parser.set_defaults()

    args = parser.parse_args()

    ########### Load data from file
    if args.verbose:
        print 'Loading data'
    
    y = np.loadtxt(args.data, delimiter=',')
    edges = np.loadtxt(args.edges, delimiter=',', dtype=int)

    if args.verbose:
        print 'Solving the GFL for {0} variables with {1} edges'.format(len(y), len(edges))
    
    
    ########### Run the C solver
    t0 = time.clock()
    beta = solve_gfl(y, edges, verbose=args.verbose)
    t1 = time.clock()

    ########### Print the timing stats
    if args.time:
        print 'Solved the GFL in {0}s and {1} total steps of ADMM.'.format(t1 - t0, np.array(solver.steps).sum())
    
    ########### Save the results to file
    if args.output:
        if args.verbose:
            print 'Saving results to {0}'.format(args.output)
        np.savetxt(args.output, beta, delimiter=',')
    else:
        print 'Results:'
        print beta

def imtv():
    parser = argparse.ArgumentParser(description='Runs the graph-fused lasso (GFL) solver on an image. Note: this is currently pretty slow for even medium-sized color images.')

    parser.add_argument('imagefile', help='The file containing the image to denoise.')
    parser.add_argument('output', help='The file to output the results.')

    parser.add_argument('--verbose', type=int, default=1, help='The level of print statements. 0=none, 1=moderate, 2=all. Default=0.')
    parser.add_argument('--time', action='store_true', help='Print the timing stats for the algorithm.')

    parser.set_defaults()

    args = parser.parse_args()

    ########### Load data from file
    if args.verbose:
        print 'Loading image'

    from scipy.misc import imsave, imread
    from utils import hypercube_edges
    
    y = imread(args.imagefile).astype(float)
    y_mean = y.mean()
    y -= y_mean

    edges = hypercube_edges(y.shape)

    if args.verbose:
        print 'Solving the GFL for {0} variables with {1} edges'.format(len(y.flatten()), len(edges))
    
    
    ########### Run the C solver
    t0 = time.clock()
    beta = solve_gfl(y.flatten(), edges, verbose=args.verbose, converge=1e-3, maxsteps=3000)
    t1 = time.clock()

    ########### Print the timing stats
    if args.time:
        print 'Solved the GFL in {0}s and {1} total steps of ADMM.'.format(t1 - t0, np.array(solver.steps).sum())
    
    ########### Save the results to file
    if args.verbose:
        print 'Saving results to {0}'.format(args.output)
    imsave(args.output, beta.reshape(y.shape) + y_mean)
    

