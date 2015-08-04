A Fast, Flexible Algorithm for the Graph-Fused Lasso
----------------------------------------------------

The goal in the graph-fused lasso (GFL) is to find a solution to the following convex optimization problem:

![GFL Convex Optimization Problem](https://raw.githubusercontent.com/tansey/gfl/master/img/eq1.png)

where __l__ is a smooth, convex loss function. The problem assumes you are given a graph structure of edges and nodes, where each node corresponds to a variable and edges between nodes correspond to constraints on the first differences between the variables. The objective function then seeks to find a solution to the above problem that minimizes the loss function over the vertices plus the sum of the first differences defined by the set of edges __E__.

The solution implemented here is based on the graph-theoretic trail decomposition and ADMM algorithm implemented in [1]. The code relies on a slightly modified version of a linear-time dynamic programming solution to the 1-d (i.e. chain) GFL [2].

Compiling the C solver lib
==========================
To compile the C solver library, you just need to run the make file from the `cpp` directory:

`make all`

Then you will need to make sure that you have the `cpp/lib` directory in your `LD_LIBRARY_PATH`:

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/path/to/gfl/cpp/lib/`

Note the above instructions are for *nix users.

Python requirements
===================
The python wrapper requires `numpy`, `scipy`, and `networkx` to be able to run everything.

Installing
==========
The package can be installed via Pip:

`pip install pygfl`

or directly from source:

```
python setup.py build
python setup.py install
```

Note that the installation has not been tested on anything other than Mac OS X and Ubuntu. The underlying solver is implemented in pure C and should be cross-platform compatible.

Running
=======
There are two steps in running the GFL solver (once installed). First, you need to decompose your graph into a set of trails then you need to run the C-based GFL solver.

#### 1) Trail decomposition
Suppose you have a graph file like the one in `example/edges.csv`, where each edge is specified on a new line, with a comma separating vertices:

```
0,1
1,2
3,4
2,4
5,4
6,0
3,6
...
```

You can then decompose this graph by running the command line `maketrails` script:

```
maketrails file --infile example/edges.csv --savet example/trails.csv
```

This will create a file in `example/trails.csv` containing a set of distinct, non-overlapping trails which minimally decomposes the original graph. Next you need to run the solver.

#### 2) Solving via ADMM
Given a set of trails in `example/trails.csv` and a vector of observations in `example/data.csv`, you can run the `graphfl` script to execute the GFL solver:

```
graphfl example/data.csv example/trails.csv --o example/smoothed.csv
```

This will run a solution path to auto-tune the value of the penalty parameter (the λ in equation 1). The results will be saved in `example/smoothed.csv`. The results should look something like the image below.

![Example GFL Solution](https://raw.githubusercontent.com/tansey/gfl/master/img/example1.png)

Licensing
=========
This library / package is distributed under the GNU Lesser General Public License, version 3. Note that a subset of code from [2] was modified and is included in the C source.

References
==========
[1] W. Tansey and J. G. Scott. "[A Fast and Flexible Algorithm for the Graph-Fused Lasso](http://arxiv.org/abs/1505.06475)," arXiv:1505.06475, May 2015.

[2] [glmgen](https://github.com/statsmaths/glmgen)