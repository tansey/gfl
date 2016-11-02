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
from scipy.sparse import issparse, isspmatrix_coo, coo_matrix
from scipy.sparse.linalg import lsqr

def create_plateaus(data, edges, plateau_size, plateau_vals, plateaus=None):
    '''Creates plateaus of constant value in the data.'''
    nodes = set(edges.keys())
    if plateaus is None:
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
            plateaus.append(set(plateau))
    for p,v in zip(plateaus, plateau_vals):
        data[np.array(list(p), dtype=int)] = v
    return plateaus

def create_tf_plateaus(k, data, edges, plateau_size, plateau_vals, plateaus=None):
    if k == 0:
        return create_plateaus(data, edges, plateau_size, plateau_vals, plateaus=plateaus)
    D = matrix_from_edges(edges)
    Dkminus1 = get_delta(D, k-1)
    if k % 2 == 0:
        plateaus = create_plateaus(data, edges, plateau_size, plateau_vals)
        data[:] = lsqr(Dkminus1, data)[0]
    else:
        # TODO: not a very principled way of creating odd-k plateaus, but it looks pretty
        p = np.zeros(Dkminus1.shape[0])
        plateaus = create_plateaus(p, edges, plateau_size, plateau_vals)
        data[:] = lsqr(Dkminus1, p)[0]
    return plateaus

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

def chains_to_trails(chains):
    rows = []
    for c in chains:
        rows.append([c[0][0]] + [x[1] for x in c])
    trails = []
    breakpoints = []
    edges = defaultdict(list)
    for nodes in rows:
        if len(trails) > 0:
            breakpoints.append(len(trails))
        trails.extend(nodes)
        for n1,n2 in zip(nodes[:-1], nodes[1:]):
            edges[n1].append(n2)
            edges[n2].append(n1)
    if len(trails) > 0:
        breakpoints.append(len(trails))
    return (len(breakpoints), np.array(trails, dtype="int32"), np.array(breakpoints, dtype="int32"), edges)
    

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

def pretty_str(p, decimal_places=2, print_zero=True, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, print_zero)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, print_zero, label_columns)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2, print_zero=True, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, print_zero) for i, a in enumerate(p)]))

def vector_str(p, decimal_places=2, print_zero=True):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if not print_zero and a == 0 else style.format(a) for a in p]))


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

def nearly_unique(arr, rel_tol=1e-4, verbose=0):
    '''Heuristic method to return the uniques within some precision in a numpy array'''
    results = np.array([arr[0]])
    for x in arr:
        if np.abs(results - x).min() > rel_tol:
            results = np.append(results, x)
    return results

def line_graph_edges(length):
    edges = defaultdict(list)
    for i in xrange(length-1):
        edges[i].append(i+1)
        edges[i+1].append(i)
    return edges

def grid_graph_edges(rows, cols):
    edges = defaultdict(list)
    for x in xrange(cols):
        for y in xrange(rows):
            if x < cols-1:
                i = int(y*cols+x)
                j = int(y*cols+x+1)
                edges[i].append(j)
                edges[j].append(i)
            if y < rows-1:
                i = int(y*cols+x)
                j = int((y+1)*cols+x)
                edges[i].append(j)
                edges[j].append(i)
    return edges

def cube_graph_edges(rows, cols, aisles):
    edges = defaultdict(list)
    for x in xrange(cols):
        for y in xrange(rows):
            for z in xrange(aisles):
                node = x * cols * aisles + y * aisles + z
                if x < cols-1:
                    i = node
                    j = (x+1) * cols * aisles + y * aisles + z
                    edges[i].append(j)
                    edges[j].append(i)
                if y < rows-1:
                    i = node
                    j = x * cols * aisles + (y+1) * aisles + z
                    edges[i].append(j)
                    edges[j].append(i)
                if z < aisles-1:
                    i = node
                    j = x * cols * aisles + y * aisles + z+1
                    edges[i].append(j)
                    edges[j].append(i)
    return edges

def hypercube_edges(dims):
    '''Create edge lists for an arbitrary hypercube. TODO: this is probably not the fasted way.'''
    edges = []
    nodes = np.arange(np.product(dims)).reshape(dims)
    for i,d in enumerate(dims):
        for j in xrange(d-1):
            for n1, n2 in zip(np.take(nodes, [j], axis=i).flatten(), np.take(nodes,[j+1], axis=i).flatten()):
                edges.append((n1,n2))
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

def get_1d_penalty_matrix(length, sparse=False):
    if sparse:
        rows = np.repeat(np.arange(length-1), 2)
        cols = np.repeat(np.arange(length), 2)[1:-1]
        data = np.tile([-1, 1], length-1)
        D = coo_matrix((data, (rows, cols)), shape=(length-1, length))
    else:
        D = np.eye(length, dtype=float)[0:-1] * -1
        for i in xrange(len(D)):
            D[i,i+1] = 1
    return D

def get_2d_penalty_matrix(rows, cols, sparse=True):
    r = 0
    rvals = []
    cvals = []
    data = []
    for y in xrange(rows):
        for x in xrange(cols - 1):
            rvals.append(r)
            rvals.append(r)
            r += 1
            cvals.append(y*cols+x)
            cvals.append(y*cols+x+1)
            data.append(-1)
            data.append(1)
    for y in xrange(rows - 1):
        for x in xrange(cols):
            rvals.append(r)
            rvals.append(r)
            r += 1
            cvals.append(y*cols+x)
            cvals.append((y+1)*cols+x)
            data.append(-1)
            data.append(1)
    D = coo_matrix((data, (rvals, cvals)), shape=(r, rows*cols))
    if not sparse:
        D = np.array(D.todense())
    return D

def special_2d(rows, cols, sparse=True):
    r = 0
    rvals = []
    cvals = []
    data = []
    for y in xrange(rows - 1):
        for x in xrange(cols):
            rvals.append(r)
            rvals.append(r)
            r += 1
            cvals.append(y*cols+x)
            cvals.append((y+1)*cols+x)
            data.append(-1)
            data.append(1)
    for y in xrange(cols - 1):
        for x in xrange(rows):
            rvals.append(r)
            rvals.append(r)
            r += 1
            cvals.append(y*rows+x+rows*cols)
            cvals.append((y+1)*rows+x+rows*cols)
            data.append(1)
            data.append(-1)
    D = coo_matrix((data, (rvals, cvals)), shape=(r, rows*cols*2))
    if not sparse:
        D = np.array(D.todense())
    return D

def get_delta(D, k):
    '''Calculate the k-th order trend filtering matrix given the oriented edge
    incidence matrix and the value of k.'''
    if k < 0:
        raise Exception('k must be at least 0th order.')
    result = D
    for i in xrange(k):
        result = D.T.dot(result) if i % 2 == 0 else D.dot(result)
    return result

def decompose_delta(deltak):
    '''Decomposes the k-th order trend filtering matrix into a c-compatible set
    of arrays.'''
    if not isspmatrix_coo(deltak):
        deltak = coo_matrix(deltak)
    dk_rows = deltak.shape[0]
    dk_rowbreaks = np.cumsum(deltak.getnnz(1), dtype="int32")
    dk_cols = deltak.col.astype('int32')
    dk_vals = deltak.data.astype('double')
    return dk_rows, dk_rowbreaks, dk_cols, dk_vals

def matrix_from_edges(edges):
    '''Returns a sparse penalty matrix (D) from a list of edge pairs. Each edge
    can have an optional weight associated with it.'''
    max_col = 0
    cols = []
    rows = []
    vals = []
    if type(edges) is defaultdict:
        edge_list = []
        for i, neighbors in edges.iteritems():
            for j in neighbors:
                if i <= j:
                    edge_list.append((i,j))
        edges = edge_list
    for i, edge in enumerate(edges):
        s, t = edge[0], edge[1]
        weight = 1 if len(edge) == 2 else edge[2]
        cols.append(min(s,t))
        cols.append(max(s,t))
        rows.append(i)
        rows.append(i)
        vals.append(weight)
        vals.append(-weight)
        if cols[-1] > max_col:
            max_col = cols[-1]
    return coo_matrix((vals, (rows, cols)), shape=(rows[-1]+1, max_col+1))

def ks_distance(a, b):
    '''Get the Kolmogorov-Smirnov (KS) distance between two densities a and b.'''
    if len(a.shape) == 1:
        return np.max(np.abs(a.cumsum() - b.cumsum()))
    return np.max(np.abs(a.cumsum(axis=1) - b.cumsum(axis=1)), axis=1)

def tv_distance(a, b):
    '''Get the Total Variation (TV) distance between two densities a and b.'''
    if len(a.shape) == 1:
        return np.sum(np.abs(a - b))
    return np.sum(np.abs(a - b), axis=1)

def edge_map_from_edge_list(edges):
    result = defaultdict(list)
    for s,t in edges:
        result[s].append(t)
        result[t].append(s)
    return result

