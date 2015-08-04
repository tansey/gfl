import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
import time
from gfl.solver import TrailSolver
from gfl.trails import *
from gfl.utils import *

def num_nodes(edges):
    s = set()
    for k,v in edges.iteritems():
        s |= set(v)
    return np.max(list(s))+1

def create_noisy_data(edges_filename):
    edges = load_edges(edges_filename)
    nnodes = num_nodes(edges)
    data = np.zeros(nnodes)
    plateau_size = max(2, nnodes / 20)
    plateau_vals = [30, 45, 10, 5]

    # Random plateaus
    create_plateaus(data, edges, plateau_size, plateau_vals)

    # Noise the data
    y = np.zeros(data.shape, dtype='double')
    y[:] = data + np.random.normal(0, 3, size=y.shape)

    return data, y

def trails_condition_number(chains):
    # Get the number of columns in the A matrix
    uniques = set()
    for chain in chains:
        for x1, x2 in chain:
            uniques.add(x1)
            uniques.add(x2)
    num_nodes = len(uniques)

    # Create the A matrix for the given chains
    # One row per z_i
    A = []
    for chain in chains:
        for node in chain:
            row = np.zeros(num_nodes)
            row[node[0]] = 1
            A.append(row)
        row = np.zeros(num_nodes)
        row[node[1]] = 1
        A.append(row)

    # Calculate the condition number of the A matrix
    A = np.array(A)
    U, s, V = np.linalg.svd(A)
    s1 = np.abs(s).max()
    sp = np.abs(s).min()
    return s1 / sp, s1, sp

def get_edgelist(filename):
    # Load the edges
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        edges = []
        for line in reader:
            edges.extend([(int(x), int(y)) for x,y in zip(line[:-1], line[1:])])
            if len(line) > 2:
                edges.append((int(line[-1]), int(line[0])))
    return edges

def scatter_results(x, steps, xlab, filename):
    plt.figure()
    plt.scatter(x, steps)
    plt.xlabel(xlab)
    plt.ylabel('# of steps until convergence')
    plt.title('Convergence Rate vs. {0}'.format(xlab))
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':
    conds = []
    sigma1 = []
    sigmap = []
    seconds = []
    steps = []
    rows = 20
    cols = 20
    edgelist = get_edgelist('../example/tempedges.csv')


    for trial in xrange(100):
        truth, y = create_noisy_data('../example/tempedges.csv')

        # Generate 100 random tour-based decompositions
        for h in ['rowcol', 'tour', 'random', 'ones']:
            for i in xrange(1):
                print 'Trial #{0}, Example #{1}'.format(trial, i)

                # Shuffle the edges so we get a different set of trails (i hope)
                np.random.shuffle(edgelist)

                if h == 'rowcol':
                    # Get the row/col split for this graph
                    ntrails, trails, breakpoints, edges = row_col_trails(rows, cols)

                    # Convert into the chain format
                    chains = []
                    for start, end in zip([0] + list(breakpoints[:-1]), breakpoints[1:]):
                        chains.append(list(zip(trails[start:end-1], trails[start+1:end])))
                else:
                    # Create the graph
                    g = nx.Graph()
                    g.add_edges_from(edgelist)

                    # Decompose the graph
                    chains = decompose_graph(g, heuristic=h)

                # Get the condition number of the A matrix
                cond, s1, sp = trails_condition_number(chains)
                conds.append(cond)
                sigma1.append(s1)
                sigmap.append(sp)

                print '\t-- Trails: {0}'.format(h)
                print '\t-- Condition number:    {0}'.format(cond)
                print '\t-- Largest eigenvalue:  {0}'.format(s1)
                print '\t-- Smallest eigenvalue: {0}'.format(sp)

                # Quick and dirty form conversion
                save_chains(chains, '../example/temptrails.csv')
                ntrails, trails, breakpoints, edges = load_trails('../example/temptrails.csv')

                # Create the solver
                solver = TrailSolver()

                # Set the data and pre-cache any necessary structures
                solver.set_data(y, edges, ntrails, trails, breakpoints)

                # Run the solver and time it
                t0 = time.clock()
                beta = solver.solve(1.0) # fix lambda arbitrarily to 1
                t1 = time.clock()

                # Get the timing results
                seconds.append(t1 - t0)
                steps.append(np.array(solver.steps).sum())

                print '\t-- Seconds: {0}'.format(seconds[-1])
                print '\t-- Steps: {0}'.format(steps[-1])

                if h in ['rowcol', 'ones']:
                    break

    print '------------ FINISHED ------------'
    steps = np.array(steps)
    conds = np.array(conds)
    sigma1 = np.array(sigma1)
    sigmap = np.array(sigmap)
    scatter_results(conds, steps, 'Condition Number', '../example/tempconds.pdf')
    scatter_results(sigma1, steps, 'Largest Eigenvalue', '../example/tempsig1.pdf')
    scatter_results(sigmap, steps, 'Smallest Eigenvalue', '../example/tempsigp.pdf')

    np.savetxt('../example/tempconds.csv', conds, delimiter=',')
    np.savetxt('../example/tempseconds.csv', seconds, delimiter=',')
    np.savetxt('../example/tempsteps.csv', steps, delimiter=',')

