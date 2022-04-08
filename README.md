A Fast, Flexible Algorithm for the Graph-Fused Lasso
----------------------------------------------------

<p align="center">
  <img src="https://github.com/tansey/gfl/blob/master/img/example1.png?raw=true" alt="Example GFL Solution"/>
</p>

The goal in the graph-fused lasso (GFL) is to find a solution to the following convex optimization problem:

<p align="center">
  <img src="https://github.com/tansey/gfl/blob/master/img/eq1.png?raw=true" alt="GFL Convex Optimization Problem"/>
</p>

where __l__ is a smooth, convex loss function. The problem assumes you are given a graph structure of edges and nodes, where each node corresponds to a variable and edges between nodes correspond to constraints on the first differences between the variables. The objective function then seeks to find a solution to the above problem that minimizes the loss function over the vertices plus the sum of the first differences defined by the set of edges __E__.

The solution implemented here is based on the graph-theoretic trail decomposition and ADMM algorithm implemented in [1]. The code relies on a slightly modified version of a linear-time dynamic programming solution to the 1-d (i.e. chain) GFL [2].

Python Requirements
===================
The python (Python version 2) wrapper requires `numpy`, `scipy`, and `networkx` to be able to run everything.
Note that the `libgraphfl` library also depends on the [Gnu Scientific Library `gsl`](https://www.gnu.org/software/gsl/) which should be available on your system.

Installing
==========
The package can be installed via Pip:

```sh
pip install pygfl
```

or directly from source:

```sh
python setup.py build
python setup.py install
```

Note that the installation has not been tested on anything other than Mac OS X and Ubuntu. The underlying solver is implemented in pure C and should be cross-platform compatible.

If you have trouble installing with pip due to linker errors with GSL and conda, you can solve this by adding your GSL headers to the include and library paths. E.g. in conda3:

```sh
conda install -c conda-forge gsl
export C_INCLUDE_PATH=/opt/anaconda3/envs/{environment name}/include/
export LIBRARY_PATH=/opt/anaconda3/envs/{environment name}/lib/
```

Running
=======
The simplest way to run the script is via the command-line `graphfl` script. You just give it a CSV of your data that you wish to smooth and a CSV of your edges, one edge per row:

```sh
graphfl example/data.csv example/edges.csv --o example/smoothed.csv
```

This will run a solution path to auto-tune the value of the penalty parameter (the λ in equation 1). The results will be saved in `example/smoothed.csv`. The results should look something like the image at the top of the readme.

Calling within Python
=====================
To call the solver within a Python program, the simplest way is to use the `easy.solve_gfl` method:

```py
import numpy as np
from pygfl.easy import solve_gfl

# Load data and edges
y = np.loadtxt('path/to/data.csv', delimiter=',')
edges = np.loadtxt('/path/to/edges.csv', delimiter=',', dtype=int)

# Run the solver
beta = solve_gfl(y, edges)
```

There are lots of other configuration options that affect the optimization procedure, but honestly they make little practical difference for most people.

Compiling the C solver lib separately
=====================================
To compile the C solver as a standalone library, you just need to run the make file from the `cpp` directory:

`make all`

Then you will need to make sure that you have the `cpp/lib` directory in your `LD_LIBRARY_PATH`:

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/my/path/to/gfl/cpp/lib/`

Note the above instructions are for \*nix users.

Licensing
=========
This library / package is distributed under the GNU Lesser General Public License, version 3. Note that a subset of code from [2] was modified and is included in the C source.

References
==========
[1] W. Tansey, O. Koyejo, R. A. Poldrack, and J. G. Scott. "[False Discovery Rate Smoothing](http://amstat.tandfonline.com/doi/abs/10.1080/01621459.2017.1319838)," Supplementary material. Journal of the American Statistical Association (JASA): Theory and Methods, June, 2017.

[2] [glmgen](https://github.com/statsmaths/glmgen)
