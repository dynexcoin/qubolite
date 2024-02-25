# qubolite using the Dynex Platform

Quantum Computing (QC) has ushered in a new era of computation, promising to solve problems that are practically infeasible for classical computers. One of the most exciting applications of quantum computing is its ability of solving combinatorial optimization problems, such as Quadratic Unconstrained Binary Optimization (QUBO). This problem class has regained significant attention with the advent of Quantum Computing. These hard-to-solve combinatorial problems appear in many different domains, including finance, logistics, Machine Learning and Data Mining. To harness the power of Quantum Computing for QUBO, The Lamarr Institute introduced qubolite, a Python package comprising utilities for creating, analyzing, and solving QUBO instances, which incorporates current research algorithms developed by scientists at the Lamarr Institute. Qubolite is a light-weight toolbox for working with QUBO instances in NumPy. This fork showcases the use of Qubolite to compute on the Dynex Neuromorphic computing platform. 

<img src="qubolite.png"  width="100" height="100">


## Installation

```
pip install qubolite
```

This package was created using Python 3.10, but runs with Python >= 3.8.

## Optional Dependencies

If you're planning to use the roof dual function as lower bound you will need to install optional
dependencies. The igraph based roof dual lower bound function can be used by calling 
`qubolite.bounds.lb_roof_dual()`. It requires that the [igraph](https://igraph.org/) library is 
installed. This can be done with `pip install igraph` or by installing qubolite with 
`pip install qubolite[roof_dual]`.

Using the function `qubolite.ordering_distance()` requires the Kendall-τ measure from the
[scipy](https://scipy.org/) library which can be installed by `pip install scipy` or by installing 
qubolite with `pip install qubolite[kendall_tau]`.

For exemplary QUBO embeddings (e.g. clustering or subset sum), the 
[scikit-learn](https://scikit-learn.org/) library is required. It can be installed by either using 
`pip install scikit-learn` or installing qubolite with `pip install qubolite[embeddings]`.

If you would like to install all optional dependencies you can use `pip install qubolite[all]` for
achieving this.

## Usage Examples

By design, `qubolite` is a shallow wrapper around `numpy` arrays, which represent QUBO parameters.
The core class is `qubo`, which receives a `numpy.ndarray` of size `(n, n)`.
Alternatively, a random instance can be created using `qubo.random()`.

```
>>> import numpy as np
>>> from qubolite import qubo
>>> arr = np.triu(np.random.random((8, 8)))
>>> Q = qubo(arr)
>>> Q2 = qubo.random(12, distr='uniform')
```

By default, `qubo()` takes an upper triangle matrix.
A non-triangular matrix is converted to an upper triangle matrix by adding the lower to the upper triangle.

To get the QUBO function value, instances can be called directly with a bit vector.
The bit vector must be a `numpy.ndarray` of size `(n,)` or `(m, n)`.

```
>>> x = np.random.random(8) < 0.5
>>> Q(x)
7.488225478498116
>>> xs = np.random.random((5,8)) < 0.5
>>> Q(xs)
array([5.81642745, 4.41380893, 11.3391062, 4.34253921, 6.07799747])
```

### Solving (Brute Force)

The submodule `solving` contains several methods to obtain the minimizing bit vector or energy value of a given QUBO instance, both exact and approximative.

```
>>> from qubolite.solving import brute_force
>>> x_min, value = brute_force(Q2)
>>> x_min
[0. 0. 0. 1. 1. 1. 1. 0. 0. 0. 1. 0.]
>>> value
-5.943917903848271
```

The method `brute_force` is implemented efficiently in C and parallelized with OpenMP.
Still, for instances with more than 30 variables take a long time to solve this way.

### Solving (Dynex Platform)

The Dynex SDK can solve Qubo problems, which are created by Qubolite, in a seamless way:

```
>>> import dynex
>>> sampleset = dynex.sample_qubo(Q2.m, mainnet=True, num_reads=1024, annealing_time=200)
[DYNEX] PRECISION SET TO 1e-05
[DYNEX] SAMPLER INITIALISED
[DYNEX|TESTNET] *** WAITING FOR READS ***
╭────────────┬─────────────┬───────────┬───────────────────────────┬─────────┬─────────┬────────────────╮
│   DYNEXJOB │   BLOCK FEE │ ELAPSED   │ WORKERS READ              │ CHIPS   │ STEPS   │ GROUND STATE   │
├────────────┼─────────────┼───────────┼───────────────────────────┼─────────┼─────────┼────────────────┤
│         -1 │           0 │           │ *** WAITING FOR READS *** │         │         │                │
╰────────────┴─────────────┴───────────┴───────────────────────────┴─────────┴─────────┴────────────────╯

[DYNEX] FINISHED READ AFTER 0.00 SECONDS
[DYNEX] SAMPLESET READY
>>> print(sampleset)
   0  1  2  3  4  5  6  7  8  9 10 11    energy num_oc.
0  0  0  0  1  1  1  1  0  0  0  1  0 -5.943918       1
['BINARY', 1 rows, 1 samples, 12 variables]
```

## Documentation

The complete API documentation can be found [here](https://smuecke.github.io/qubolite/api.html).

## Version Log

* **0.2** Added problem embeddings (binary clustering, subset sum problem)
* **0.3** Added `QUBOSample` class and sampling methods `full` and `gibbs`
* **0.4** Renamed `QUBOSample` to `BinarySample`; added methods for saving and loading QUBO and Sample instances
* **0.5** Moved `gibbs` to `mcmc` and implemented true Gibbs sampling as `gibbs`; added `numba` as dependency
    * **0.5.1** changed `keep_prob` to `keep_interval` in Gibbs sampling, making the algorithm's runtime deterministic; renamed `sample` to `random` in QUBO embedding classes, added MAX 2-SAT problem embedding
* **0.6** Changed Python version to 3.8; removed `bitvec` dependency; added `scipy` dependency required for matrix operations in numba functions
    * **0.6.1** added scaling and rounding
    * **0.6.2** removed `seedpy` dependency
    * **0.6.3** renamed `shots` to `size` in `BinarySample`; cleaned up sampling, simplified type hints
    * **0.6.4** added probabilistic functions to `qubo` class
    * **0.6.5** complete empirical prob. vector can be returned from `BinarySample`
    * **0.6.6** fixed spectral gap implementation
    * **0.6.7** moved `brute_force` to new sub-module `solving`; added some approximate solving methods
    * **0.6.8** added `bitvec` sub-module; `dynamic_range` now uses bits by default, changed `bits=False` to `decibel=False`; removed scipy from requirements
    * **0.6.9** new, more memory-efficient save format
    * **0.6.10** fixed requirements in `setup.py`; fixed size estimation in `qubo.save()`
* **0.7** Added more efficient brute-force implementation using C extension; added optional dependencies for calculating bounds and ordering distance
* **0.8** New embeddings, new solving methods; switched to NumPy random generators from `RandomState`; added parameter compression for dynamic range reduction; Added documentation
    * **0.8.1** some fixes to documentation
    * **0.8.2** implemented `qubo.dx2()`; added several new solving heuristics
    * **0.8.3** added submodule `preprocessing` and moved DR reduction there; added `partial_assignment` class as replacement of `qubo.clamp()`, which is now deprecated
    * **0.8.4** added fast Gibbs sampling and QUBO parameter training
