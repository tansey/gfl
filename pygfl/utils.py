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
import os
import argparse
import csv
import datetime
from collections import defaultdict, deque
from scipy.sparse import issparse, isspmatrix_coo

def create_plateaus(data, edges, plateau_size, plateau_vals):
    nodes = set(edges.keys())
    plateaus = []
    for i in xrange(len(plateau_vals)):
        if len(nodes) == 0:
            break
        node = np.random.choice(list(nodes))
        nodes.remove(node)
        plateau = [node]
        available = set(edges[node]) & nodes
        while len(nodes) > 0 and len(available) > 0 and len(plateau) < plateau_size:
            node = np.random.choice(list(available))
            plateau.append(node)
            available |= nodes & set(edges[node])
            available.remove(node)
        nodes -= set(plateau)
        plateaus.append(plateau)
    for p,v in zip(plateaus, plateau_vals):
        data[p] = v

def load_trails(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        trails = []
        breakpoints = []
        edges = defaultdict(list)
        for line in reader:
            if len(trails) > 0:
                breakpoints.append(len(trails))
            nodes = [int(x) for x in line]
            trails.extend(nodes)
            for n1,n2 in zip(nodes[:-1], nodes[1:]):
                edges[n1].append(n2)
                edges[n2].append(n1)
        if len(trails) > 0:
            breakpoints.append(len(trails))
    return (len(breakpoints), np.array(trails, dtype="int32"), np.array(breakpoints, dtype="int32"), edges)

def save_trails(filename, trails, breakpoints):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for start, stop in zip([0]+list(breakpoints[:-1]), breakpoints):
            writer.writerow(trails[start:stop])

def load_edges(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        edges = defaultdict(list)
        for line in reader:
            nodes = [int(x) for x in line]
            for n1,n2 in zip(nodes[:-1], nodes[1:]):
                edges[n1].append(n2)
                edges[n2].append(n1)
    return edges

def num_edges(edges):
    return sum([len(y) for x,y in edges.iteritems()]) / 2

def save_edges(filename, edges):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for x,Y in edges.iteritems():
            for y in Y:
                if x < y:
                    writer.writerow((x,y))

def sparse_matrix_to_edges(data):
    if not isspmatrix_coo(data):
        data = data.tocoo()
    edges = defaultdict(list)
    for x,y in zip(data.row, data.col):
        if x != y and y not in edges[x]:
            edges[x].append(y)
            edges[y].append(x)
    return edges

def pretty_str(p, decimal_places=2):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([vector_str(a, decimal_places) for a in p]))

def vector_str(p, decimal_places=2):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([style.format(a) for a in p]))


def make_directory(base, subdir):
    if not base.endswith('/'):
        base += '/'
    directory = base + subdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not directory.endswith('/'):
        directory = directory + '/'
    return directory

def calc_plateaus(beta, edges, rel_tol=1e-4, verbose=0):
    '''Calculate the plateaus (degrees of freedom) of a graph of beta values in linear time.'''
    to_check = deque(xrange(len(beta)))
    check_map = np.zeros(beta.shape, dtype=bool)
    check_map[np.isnan(beta)] = True
    plateaus = []

    if verbose:
        print '\tCalculating plateaus...'

    if verbose > 1:
        print '\tIndices to check {0} {1}'.format(len(to_check), check_map.shape)

    # Loop until every beta index has been checked
    while to_check:
        if verbose > 1:
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
        min_member = val - rel_tol
        max_member = val + rel_tol

        # Check every possible boundary of the plateau
        while cur_unchecked:
            idx = cur_unchecked.popleft()
            
            # neighbors to check
            local_check = []

            # Generic graph case, get all neighbors of this node
            local_check.extend(edges[idx])

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

def grid_graph_edges(rows, cols):
    edges = []
    for x in xrange(cols):
        for y in xrange(rows):
            if x < cols-1:
                edges.append((y*cols+x,y*cols+x+1))
            if y < rows-1:
                edges.append((y*cols+x,(y+1)*cols+x))
    return edges

def cube_graph_edges(rows, cols, aisles):
    edges = []
    for x in xrange(cols):
        for y in xrange(rows):
            for z in xrange(aisles):
                node = x * cols * aisles + y * aisles + z
                if x < cols-1:
                    edges.append((node, (x+1) * cols * aisles + y * aisles + z))
                if y < rows-1:
                    edges.append((node, x * cols * aisles + (y+1) * aisles + z))
                if z < aisles-1:
                    edges.append((node, x * cols * aisles + y * aisles + z+1))
    return edges

def row_col_trails(rows, cols):
    nnodes = rows * cols
    ntrails = rows + cols
    trails = np.zeros(nnodes * 2, dtype='int32')
    trails[:nnodes] = np.arange(nnodes, dtype='int32') # row trails
    trails[nnodes:] = np.arange(nnodes).reshape((rows, cols)).T.flatten() # column trails
    breakpoints = np.zeros(ntrails, dtype='int32')
    breakpoints[:rows] = np.arange(1, rows+1) * cols
    breakpoints[rows:] = nnodes + np.arange(1, cols+1) * rows
    return ntrails, trails, breakpoints, grid_graph_edges(rows, cols)







