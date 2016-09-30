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
import csv
import argparse
import networkx as nx
from itertools import combinations
from collections import defaultdict

def plot_trails(subgraphs, trails, step):
    import matplotlib.pyplot as plt
    subgraphs_and_trails = subgraphs + trails
    max_cols = 4
    rows = int(len(subgraphs_and_trails) / max_cols) + min(1, len(subgraphs_and_trails) % max_cols)
    cols = min(len(subgraphs_and_trails), max_cols)
    fig, axarr = plt.subplots(rows, cols, figsize=(cols*5, rows*5+1))
    for i, g in enumerate(subgraphs_and_trails):
        ax = axarr if cols == 1 else (axarr[int(i / max_cols), i % max_cols] if rows > 1 else axarr[i])
        if i >= len(subgraphs):
            trail_g = nx.Graph()
            trail_g.add_edges_from(g)
            g = trail_g
        nx.draw(g, ax=ax, with_labels=True)
        if i < len(subgraphs):
            ax.set_title('Unvisited subgraph #{0}'.format(i+1))
        else:
            ax.set_title('Trail #{0}'.format(i+1-len(subgraphs)))
    if rows > 1 and len(subgraphs_and_trails) % max_cols > 0:
        for i in xrange((len(subgraphs_and_trails) % max_cols), max_cols):
            ax = axarr if cols == 1 else (axarr[rows-1, i] if rows > 1 else axarr[i])
            ax.axis('off')
    plt.savefig('../img/graph{0:05d}.pdf'.format(step))
    plt.clf()

def calc_euler_tour(g, start, end):
    '''Calculates an Euler tour over the graph g from vertex start to vertex end.
    Assumes start and end are odd-degree vertices and that there are no other odd-degree
    vertices.'''
    even_g = nx.subgraph(g, g.nodes())
    if end in even_g.neighbors(start):
        # If start and end are neighbors, remove the edge
        even_g.remove_edge(start, end)
        comps = list(nx.connected_components(even_g))
        # If the graph did not split, just find the euler circuit
        if len(comps) == 1:
            trail = list(nx.eulerian_circuit(even_g, start))
            trail.append((start, end))
        elif len(comps) == 2:
            subg1 = nx.subgraph(even_g, comps[0])
            subg2 = nx.subgraph(even_g, comps[1])
            start_subg, end_subg = (subg1, subg2) if start in subg1.nodes() else (subg2, subg1)
            trail = list(nx.eulerian_circuit(start_subg, start)) + [(start, end)] + list(nx.eulerian_circuit(end_subg, end))
        else:
            raise Exception('Unknown edge case with connected components of size {0}:\n{1}'.format(len(comps), comps))
    else:
        # If they are not neighbors, we add an imaginary edge and calculate the euler circuit
        even_g.add_edge(start, end)
        circ = list(nx.eulerian_circuit(even_g, start))
        try:
            trail_start = circ.index((start, end))
        except:
            trail_start = circ.index((end, start))
        trail = circ[trail_start+1:] + circ[:trail_start]
    return trail

def random_graph_sparsity(num_nodes, sparsity=0.98, min_edges=30):
    # Create a randomly connected graph
    g = nx.Graph()
    if num_nodes < 1000:
        edges = np.where(np.random.random(size=(num_nodes,num_nodes)) >= sparsity)
        edges = [(v1,v2) for v1, v2 in zip(edges[0], edges[1]) if v1 != v2]
    else:
        # Handle huge graphs
        edges = [(v1,v2) for v1, v2 in np.random.choice(np.arange(num_nodes), size=(max(min_edges, num_nodes*num_nodes*(1-sparsity)), 2)) if v1 != v2]
    # Enforce a minimum amount of edges
    while len(edges) < min_edges:
        v1, v2 = np.random.choice(np.arange(num_nodes), size=2)
        if (v1, v2) not in edges:
            edges.append((v1, v2))
    g.add_edges_from(edges)
    return g

def random_graph_edges(num_nodes, num_edges):
    # Create a randomly connected graph
    g = nx.Graph()
    edges = [(v1,v2) for v1, v2 in np.random.choice(np.arange(num_nodes), size=(num_edges, 2)) if v1 != v2]
    g.add_edges_from(edges)
    
    # If the random graph is not fully connected, connect all the subgraphs
    subgraphs = [nx.subgraph(g, x) for x in nx.connected_components(g)]
    while len(subgraphs) > 1:
        for i,subg in enumerate(subgraphs):
            v1 = np.random.choice(subg.nodes())
            subg2 = subgraphs[np.random.choice(np.delete(np.arange(len(subgraphs)), i))]
            v2 = np.random.choice(subg2.nodes())
            g.add_edge(v1, v2)
        subgraphs = [nx.subgraph(g, x) for x in nx.connected_components(g)]

    return g


def bowtie_graph():
    bowtie = [('A', 'B'),('A', 'C'),('B', 'C'),('A', 'D'),('D', 'E'), ('D', 'F'), ('E', 'F')]
    g = nx.Graph()
    g.add_edges_from(bowtie)
    return g

def grid_graph(rows, cols):
    edges = []
    for x in xrange(cols):
        for y in xrange(rows):
            if x < cols-1:
                edges.append((y*cols+x,y*cols+x+1))
            if y < rows-1:
                edges.append((y*cols+x,(y+1)*cols+x))
    g = nx.Graph()
    g.add_edges_from(edges)
    return g

def cube_graph(rows, cols, aisles):
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
    g = nx.Graph()
    g.add_edges_from(edges)
    return g

def save_graph(g, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for x,y in g.edges():
            writer.writerow([x,y])

def load_graph(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        edges = []
        for line in reader:
            edges.extend([(int(x), int(y)) for x,y in zip(line[:-1], line[1:])])
            if len(line) > 2:
                edges.append((int(line[-1]), int(line[0])))
        g = nx.Graph()
        g.add_edges_from(edges)
    return g

def save_chains(trails, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for t in trails:
            row = [t[0][0]] + [x[1] for x in t]
            writer.writerow(row)

def select_odd_degree_trail(subg, odds, max_odds, heuristic, verbose):
    # Otherwise, we need to remove a trail from one odd-degree to another
    paths = []
    path_lengths = []
    if len(odds) <= max_odds:
        # Find the shortest path between each possible pair
        for j,x in enumerate(odds):
            for y in odds[j+1:]:
                p = nx.shortest_path(subg, x, y)
                paths.append(p)
                path_lengths.append(len(p))
    else:
        if verbose > 2:
            print '\t\t\tSampling {0} paths'.format(int(max_odds**2 / 2))
        for x, y in np.random.choice(odds, size=(int(max_odds**2 / 2), 2)):
            if verbose > 3:
                print '\t\t\t\tSample {0}'.format(len(paths))
            if x == y:
                continue
            p = nx.shortest_path(subg, x, y)
            paths.append(p)
            path_lengths.append(len(p))
    path_lengths = np.array(path_lengths)

    # Choose the median
    if heuristic == 'median':
        median = np.median(path_lengths)
        trailidx = np.argmin(np.abs(path_lengths - median))
    # Choose the longest
    elif heuristic == 'max':
        trailidx = np.argmax(path_lengths)
    # Choose the shortest
    elif heuristic == 'min':
        trailidx = np.argmin(path_lengths)
    # Choose a random trail
    elif heuristic == 'any':
        trailidx = np.random.randint(0, len(paths))
    else:
        raise Exception('Unknown heuristic: {0}'.format(heuristic))

    return [path_to_trail(paths[trailidx])]

def path_to_trail(path):
    return list(zip(path[:-1],path[1:]))

def sample_pairs(nodes, num_samples):
    result = set()
    while len(result) < num_samples:
        pair = tuple(np.random.choice(nodes, 2, replace=False))
        if pair not in result:
            result.add(pair)
    return result

def select_min_degree_trail(subg, max_nodes, verbose):
    min_degree = np.array(nx.degree(subg).values()).min()
    nodes = [n for n,d in nx.degree(subg).iteritems() if d == min_degree]

    # Ugly quick and dirty... grossssss
    while len(nodes) < 2:
        min_degree += 1
        nodes.extend([n for n,d in nx.degree(subg).iteritems() if d == min_degree])
    
    if args.verbose > 1:
        print '\t\tMin degree: {0}. # min nodes: {1}'.format(min_degree, len(nodes))

    pairs = combinations(nodes, 2) if len(nodes) < max_nodes else sample_pairs(nodes, max_nodes * (max_nodes-1))
    paths = [nx.shortest_path(subg, x, y) for x,y in pairs]
    sizes = np.array([len(p) for p in paths])
    median = np.median(sizes)
    trailidx = np.argmax(np.abs(sizes - median))
    print trailidx, sizes[trailidx]
    return [path_to_trail(paths[trailidx])]


def select_random_trail(subg, verbose):
    trail = list()
    while True:
        start = trail[-1][1] if len(trail) > 0 else np.random.choice(subg.nodes())
        options = [(start, n) for n in subg.neighbors(start) if ((start,n) not in trail and (n,start) not in trail)]
        if len(options) > 0:
            trail.append(options[np.random.choice(len(options))])
            continue
        return [trail]

def select_single_edge_trails(subg, verbose):
    return [[(x,y)] for x,y in subg.edges()]

def pseudo_tour_trails(subg, odds, verbose):
    # create a working copy of the subgraph
    g = nx.Graph()
    g.add_edges_from(subg.edges())

    # Create a graph with imaginary edges between odd-degree nodes
    pseudo_edges = []
    while len(odds) > 2:
        np.random.shuffle(odds)
        v1 = odds[0]
        neighbors = g.neighbors(v1)
        v2 = None
        for v in odds[1:]:
            if v not in neighbors:
                v2 = v
                break
        odds.remove(v1)
        if v2 is None:
            # Handle the edge case where V1 is a hub to all odd-degree edges
            odds.append(v1)
        else:
            pseudo_edges.append((v1,v2))
            odds.remove(v2)
            g.add_edge(v1,v2)

    # Calculate the Eulerian tour over the pseudo-graph
    bigtrail = calc_euler_tour(g, odds[0], odds[1])

    # Remove the pseudo-edges and treat them as the trail breakpoints
    pseudo_edges = set(pseudo_edges)
    trails = []
    start = 0
    for i, (v1, v2) in enumerate(bigtrail):
        if (v1, v2) in pseudo_edges or (v2, v1) in pseudo_edges:
            trails.append(bigtrail[start:i])
            start = i+1

    # If we didn't end on an imaginary edge, we need to add the last trail
    if start < len(bigtrail):
        trails.append(bigtrail[start:])

    # Return the n trails for a graph with 2n odd-degree vertices
    return trails

def greedy_trails(subg, odds, verbose):
    '''Greedily select trails by making the longest you can until the end'''
    if verbose:
        print '\tCreating edge map'

    edges = defaultdict(list)

    for x,y in subg.edges():
        edges[x].append(y)
        edges[y].append(x)

    if verbose:
        print '\tSelecting trails'

    trails = []
    for x in subg.nodes():
        if verbose > 2:
            print '\t\tNode {0}'.format(x)

        while len(edges[x]) > 0:
            y = edges[x][0]
            trail = [(x,y)]
            edges[x].remove(y)
            edges[y].remove(x)
            while len(edges[y]) > 0:
                x = y
                y = edges[y][0]
                trail.append((x,y))
                edges[x].remove(y)
                edges[y].remove(x)
            trails.append(trail)
    return trails


def decompose_graph(g, heuristic='tour', max_odds=20, verbose=0):
    '''Decompose a graph into a set of non-overlapping trails.'''
    # Get the connected subgraphs
    subgraphs = [nx.subgraph(g, x) for x in nx.connected_components(g)]

    chains = []
    num_subgraphs = len(subgraphs)
    step = 0
    while num_subgraphs > 0:
        if verbose:
            print 'Step #{0} ({1} subgraphs)'.format(step, num_subgraphs)

        for i in xrange(num_subgraphs-1, -1, -1):
            subg = subgraphs[i]

            # Get all odd-degree nodes
            odds = [x for x,y in nx.degree(subg).iteritems() if y % 2 == 1]

            if verbose > 1:
                if len(odds) == 0:
                    print '\t\tNo odds'
                elif len(odds) == 2:
                    print '\t\tExactly 2 odds'
                else:
                    print '\t\t{0} odds'.format(len(odds))
            
            # If there are no odd-degree edges, we can find an euler circuit
            if len(odds) == 0:
                trails = [list(nx.eulerian_circuit(subg))]
            elif len(odds) == 2:
                # If there are only two odd-degree edges, we can find an euler tour
                trails = [calc_euler_tour(subg, odds[0], odds[1])]
            elif heuristic in ['min', 'max', 'median', 'any']:
                trails = select_odd_degree_trail(subg, odds, max_odds, heuristic, verbose)
            elif heuristic == 'random':
                trails = select_random_trail(subg, verbose)
            elif heuristic == 'mindegree':
                trails = select_min_degree_trail(subg, max_odds, verbose)
            elif heuristic == 'ones':
                trails = select_single_edge_trails(subg, verbose)
            elif heuristic == 'tour':
                trails = pseudo_tour_trails(subg, odds, verbose)
            elif heuristic == 'greedy':
                trails = greedy_trails(subg, odds, verbose)

            if verbose > 2:
                print '\t\tTrails: {0}'.format(len(trails))

            # Remove the trail
            for trail in trails:
                subg.remove_edges_from(trail)

            # Add it to the list of chains
            chains.extend(trails)
            
            # If the subgraph is empty, remove it from the list
            if subg.number_of_edges() == 0:
                del subgraphs[i]
            else:
                comps = list(nx.connected_components(subg))

                # If the last edge split the graph, add the new subgraphs to the list of subgraphs
                if len(comps) > 1:
                    for x in comps:
                        compg = nx.subgraph(subg, x)
                        if compg.number_of_edges() > 0:
                            subgraphs.append(compg)
                    del subgraphs[i]

        # Update the count of connected subgraphs
        num_subgraphs = len(subgraphs)
        step += 1

    return chains

def main():
    parser = argparse.ArgumentParser(description='Decomposes a graph into a minimal set of trails.')

    parser.add_argument('graph_type', choices=['file', 'random', 'grid', 'cube', 'f', 'r', 'g', 'c'], help='How to create the graph.')
    parser.add_argument('--saveg', help='Save the full graph to the specified file in CSV format.')
    parser.add_argument('--savet', help='Save the resulting set of trails to the specified file in CSV format.')
    parser.add_argument('--verbose', type=int, default=1, help='Print detailed progress information to the console. 0=none, 1=high-level only, 2=all details.')
    
    # Settings to load the graph from file
    parser.add_argument('--infile', help='Load the graph from a CSV file of edges.')

    # Generate random graph settings
    parser.add_argument('--n', type=int, default=100, help='The number of nodes in a randomly generated graph.')
    #parser.add_argument('--sparsity', type=float, default=0.99998, help='The level of sparsity in the graph.')
    #parser.add_argument('--min_edges', type=int, default=30, help='The minimum number of edges per graph.')
    parser.add_argument('--num_edges', type=int, help='The number of edges in the randomly generated graph.')


    # Generate grid and cube graph settings
    parser.add_argument('--rows', type=int, default=10, help='The number of rows in the grid or cube graph.')
    parser.add_argument('--cols', type=int, default=10, help='The number of columns in the grid or cube graph.')
    parser.add_argument('--aisles', type=int, default=10, help='The number of aisles (3rd dimension version of a row/col) in the cube graph.' )

    parser.add_argument('--max_odds', type=int, default=20, help='The maximum number of odd-degree edges to enumerate all n^2 possibilities before reverting to sampling.')
    parser.add_argument('--plot', action='store_true', help='Plot each individual step of the algorithm.')
    parser.add_argument('--heuristic', choices=['min', 'max', 'median', 'random', 'any', 'mindegree', 'ones', 'tour', 'greedy'], default='tour', help='The trail selection heuristic.')

    # TODO: Add minimum trail count option that repeatedly splits the longest path until satisfied.

    parser.set_defaults(plot=False)

    # Get the arguments from the command line
    args = parser.parse_args()

    if args.graph_type == 'file' or args.graph_type == 'f':
        if args.verbose:
            print 'Loading graph from {0}'.format(args.infile)
        g = load_graph(args.infile)
    elif args.graph_type == 'grid' or args.graph_type == 'g':
        if args.verbose:
            print 'Creating a {0} x {1} grid graph'.format(args.rows, args.cols)
        g = grid_graph(args.rows, args.cols)
    elif args.graph_type == 'cube' or args.graph_type == 'c':
        if args.verbose:
            print 'Creating a {0} x {1} x {2} cube graph'.format(args.rows, args.cols, args.aisles)
        g = cube_graph(args.rows, args.cols, args.aisles)
    elif args.graph_type == 'random' or args.graph_type == 'r':
        if args.verbose:
            #print 'Creating a random graph with {0} nodes and sparsity of {1}'.format(args.n, args.sparsity)
            print 'Creating a random graph with {0} nodes and {1} edges'.format(args.n, args.num_edges)
        #g = random_graph(args.n, sparsity=args.sparsity, min_edges=args.min_edges)
        g = random_graph_edges(args.n, args.num_edges)
    else:
        raise Exception('Unknown graph type: {0}'.format(args.graph_type))

    total_edges = g.number_of_edges() # sanity check number at the end

    if args.verbose:
        print '# of nodes in graph: {0}'.format(g.number_of_nodes())
        print '# of edges in graph: {0}'.format(total_edges)

    
    if args.saveg:
        if args.verbose:
            print 'Saving graph to {0}'.format(args.saveg)
        save_graph(g, args.saveg)

    # Plot the initial full graph
    # if args.plot:
    #     if args.verbose:
    #         print 'Plotting initial graph'
    #     fig, ax = plt.subplots()
    #     nx.draw(g, with_labels=True, ax=ax)
    #     ax.set_title('Full Graph')
    #     plt.savefig('../img/graph{0:05d}.pdf'.format(0))
    #     plt.clf()

    # Run the graph decomposition algorithm and get the resulting trails
    chains = decompose_graph(g, heuristic=args.heuristic, max_odds=args.max_odds, verbose=args.verbose)
        
    # if args.plot:
    #     if args.verbose > 1:
    #         '\tPlotting trails'
    #     plot_trails(subgraphs, chains, step)

    # Sanity check the trails to make sure we found all the edges
    if args.verbose:
        print 'Verifying solution... (will crash if failed)'
    count = sum([len(t) for t in chains])
    assert(count == total_edges)

    v = np.array([len(t) for t in chains])
    print 'mean trail length: {0} stdev: {1} min: {2} max: {3}'.format(v.mean(0), v.std(), v.min(), v.max())

    if args.savet:
        if args.verbose:
            print 'Saving trails to {0}'.format(args.savet)
        save_chains(chains, args.savet)
