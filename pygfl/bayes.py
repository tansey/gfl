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
import numpy as np
import matplotlib.pylab as plt
from numpy.ctypeslib import ndpointer
from ctypes import *
from utils import *

'''Load the bayesian GFL library'''
gflbayes_lib = cdll.LoadLibrary('libgraphfl.so')
gflbayes_gaussian_laplace = gflbayes_lib.bayes_gfl_gaussian_laplace
gflbayes_gaussian_laplace.restype = None
gflbayes_gaussian_laplace.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_gaussian_laplace_gamma = gflbayes_lib.bayes_gfl_gaussian_laplace_gamma
gflbayes_gaussian_laplace_gamma.restype = None
gflbayes_gaussian_laplace_gamma.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_gaussian_laplace_gamma_robust = gflbayes_lib.bayes_gfl_gaussian_laplace_gamma_robust
gflbayes_gaussian_laplace_gamma_robust.restype = None
gflbayes_gaussian_laplace_gamma_robust.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_double,
                c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_gaussian_doublepareto = gflbayes_lib.bayes_gfl_gaussian_doublepareto
gflbayes_gaussian_doublepareto.restype = None
gflbayes_gaussian_doublepareto.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_double, c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_gaussian_doublepareto2 = gflbayes_lib.bayes_gfl_gaussian_doublepareto2
gflbayes_gaussian_doublepareto2.restype = None
gflbayes_gaussian_doublepareto2.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_gaussian_cauchy = gflbayes_lib.bayes_gfl_gaussian_cauchy
gflbayes_gaussian_cauchy.restype = None
gflbayes_gaussian_cauchy.argtypes = [c_int, ndpointer(c_double, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_binomial_laplace = gflbayes_lib.bayes_gfl_binomial_laplace
gflbayes_binomial_laplace.restype = None
gflbayes_binomial_laplace.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_binomial_laplace_gamma = gflbayes_lib.bayes_gfl_binomial_laplace_gamma
gflbayes_binomial_laplace_gamma.restype = None
gflbayes_binomial_laplace_gamma.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_empirical_binomial_laplace_gamma = gflbayes_lib.empirical_bayes_gfl_binomial_laplace_gamma
gflbayes_empirical_binomial_laplace_gamma.restype = None
gflbayes_empirical_binomial_laplace_gamma.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]


gflbayes_binomial_doublepareto = gflbayes_lib.bayes_gfl_binomial_doublepareto
gflbayes_binomial_doublepareto.restype = None
gflbayes_binomial_doublepareto.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_double, c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_poisson_laplace = gflbayes_lib.bayes_gfl_poisson_laplace
gflbayes_poisson_laplace.restype = None
gflbayes_poisson_laplace.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

gflbayes_poisson_doublepareto = gflbayes_lib.bayes_gfl_poisson_doublepareto
gflbayes_poisson_doublepareto.restype = None
gflbayes_poisson_doublepareto.argtypes = [c_int, ndpointer(c_int, flags='C_CONTIGUOUS'),
                c_int, ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_int, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS'),
                c_double, c_double,
                c_double, c_double, c_double,
                c_long, c_long, c_long,
                ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS'), ndpointer(c_double, flags='C_CONTIGUOUS')]

trunc_norm = gflbayes_lib.rnorm_trunc_norand
trunc_norm.restype = c_double
trunc_norm.argtypes = [c_double, c_double, c_double, c_double]

def double_matrix_to_c_pointer(x):
    return (x.__array_interface__['data'][0] + np.arange(x.shape[0])*x.strides[0]).astype(np.uintp)

def sample_gtf(data, D, k, likelihood='gaussian', prior='laplace',
                           lambda_hyperparams=None, lam_walk_stdev=0.01, lam0=1.,
                           dp_hyperparameter=None, w_hyperparameters=None,
                           iterations=7000, burn=2000, thin=10,
                           robust=False, empirical=False,
                           verbose=False):
    '''Generate samples from the generalized graph trend filtering distribution via a modified Swendsen-Wang slice sampling algorithm.
    Options for likelihood: gaussian, binomial, poisson. Options for prior: laplace, doublepareto.'''
    Dk = get_delta(D, k)
    dk_rows, dk_rowbreaks, dk_cols, dk_vals = decompose_delta(Dk)

    if likelihood == 'gaussian':
        y, w = data
    elif likelihood == 'binomial':
        trials, successes = data
    elif likelihood == 'poisson':
        obs = data
    else:
        raise Exception('Unknown likelihood type: {0}'.format(likelihood))

    if prior == 'laplace':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1., 1.)
    elif prior == 'laplacegamma':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1., 1.)
        if dp_hyperparameter == None:
            dp_hyperparameter = 1.
    elif prior == 'doublepareto' or prior == 'doublepareto2':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1.0, 1.0)
        if dp_hyperparameter == None:
            dp_hyperparameter = 0.1
    elif prior == 'cauchy':
        if lambda_hyperparams == None:
            lambda_hyperparams = (1.0, 1.0)
    else:
        raise Exception('Unknown prior type: {0}.'.format(prior))

    if robust and w_hyperparameters is None:
        w_hyperparameters = (1., 1.)

    # Run the Gibbs sampler
    sample_size = (iterations - burn) / thin
    beta_samples = np.zeros((sample_size, D.shape[1]), dtype='double')
    lam_samples = np.zeros(sample_size, dtype='double')

    if likelihood == 'gaussian':
        if prior == 'laplace':
            gflbayes_gaussian_laplace(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'laplacegamma':
            if robust:
                gflbayes_gaussian_laplace_gamma_robust(len(y), y, w,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lambda_hyperparams[0], lambda_hyperparams[1],
                                          dp_hyperparameter,
                                          w_hyperparameters[0], w_hyperparameters[1],
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
            else:    
                gflbayes_gaussian_laplace_gamma(len(y), y, w,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lambda_hyperparams[0], lambda_hyperparams[1],
                                          dp_hyperparameter,
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto':
            gflbayes_gaussian_doublepareto(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0, dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto2':
            gflbayes_gaussian_doublepareto2(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'cauchy':
            gflbayes_gaussian_cauchy(len(y), y, w,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
    elif likelihood == 'binomial':
        if prior == 'laplace':
            gflbayes_binomial_laplace(len(trials), trials, successes,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto':
            gflbayes_binomial_doublepareto(len(trials), trials, successes,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0, dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'laplacegamma':
            if empirical:
                gflbayes_empirical_binomial_laplace_gamma(len(trials), trials, successes,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lam0,
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
            else:
                gflbayes_binomial_laplace_gamma(len(trials), trials, successes,
                                          dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                          lambda_hyperparams[0], lambda_hyperparams[1],
                                          dp_hyperparameter,
                                          iterations, burn, thin,
                                          double_matrix_to_c_pointer(beta_samples), lam_samples)
    elif likelihood == 'poisson':
        if prior == 'laplace':
            gflbayes_poisson_laplace(len(obs), obs,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
        elif prior == 'doublepareto':
            gflbayes_poisson_doublepareto(len(obs), obs,
                                      dk_rows, dk_rowbreaks, dk_cols, dk_vals,
                                      lambda_hyperparams[0], lambda_hyperparams[1],
                                      lam_walk_stdev, lam0, dp_hyperparameter,
                                      iterations, burn, thin,
                                      double_matrix_to_c_pointer(beta_samples), lam_samples)
    else:
        raise Exception('Unknown likelihood type: {0}'.format(likelihood))

    return (beta_samples,lam_samples)

def test_sample_gtf_poisson():
    probs = np.zeros(100)
    probs[:25] = 5.
    probs[25:50] = 9.
    probs[50:75] = 3.
    probs[75:] = 6.
    obs = np.array([np.random.poisson(p) for p in probs]).astype('int32')
    D = get_1d_penalty_matrix(len(obs))
    k = 0

    z_samples, lam_samples = sample_gtf(obs, D, k, likelihood='poisson', prior='doublepareto', verbose=True, iterations=15000, burn=2000, thin=10)
    z = z_samples.mean(axis=0)
    z_stdev = z_samples.std(axis=0)
    z_lower = z - z_stdev*2
    z_upper = z + z_stdev*2

    fig, ax = plt.subplots(4)
    x = np.linspace(0,1,len(obs))
    ax[0].bar(x, obs, width=1./len(x), color='darkblue', alpha=0.3)
    ax[0].set_xlim([0,1])
    ax[0].set_ylabel('Observations')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,12]) / (np.arange(z_samples.shape[0])+1.), color='orange')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,37]) / (np.arange(z_samples.shape[0])+1.), color='skyblue')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,63]) / (np.arange(z_samples.shape[0])+1.), color='black')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,87]) / (np.arange(z_samples.shape[0])+1.), color='#009E73')
    ax[1].set_xlim([1,z_samples.shape[0]])
    ax[1].set_ylabel('Mean values')
    ax[2].hist(lam_samples, 50)
    ax[2].set_xlabel('Lambda values')
    ax[2].set_ylabel('Lambda')
    ax[3].scatter(x, probs, alpha=0.5)
    ax[3].plot(x, z, lw=2, color='orange')
    ax[3].fill_between(x, z_lower, z_upper, alpha=0.3, color='orange')
    ax[3].set_xlim([0,1])
    ax[3].set_ylabel('Probability of success')
    plt.show()
    plt.clf()

def test_sample_gtf_binomial():
    trials = np.random.randint(5, 30, size=100).astype('int32')
    probs = np.zeros(100)
    probs[:25] = 0.25
    probs[25:50] = 0.75
    probs[50:75] = 0.5
    probs[75:] = 0.1
    successes = np.array([np.random.binomial(t, p) for t,p in zip(trials, probs)], dtype='int32')
    
    D = get_1d_penalty_matrix(len(successes))
    k = 0

    z_samples, lam_samples = sample_gtf((trials, successes), D, k, likelihood='binomial', prior='laplacegamma', verbose=True)
    z = z_samples.mean(axis=0)
    z_stdev = z_samples.std(axis=0)
    z_lower = z - z_stdev*2
    z_upper = z + z_stdev*2

    fig, ax = plt.subplots(4)
    x = np.linspace(0,1,len(trials))
    ax[0].bar(x, successes, width=1./len(x), color='darkblue', alpha=0.3)
    ax[0].bar(x, trials-successes, width=1./len(x), color='skyblue', alpha=0.3, bottom=successes)
    ax[0].set_ylim([0,30])
    ax[0].set_xlim([0,1])
    ax[0].set_ylabel('Trials and successes')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,12]) / (np.arange(z_samples.shape[0])+1.), color='orange')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,37]) / (np.arange(z_samples.shape[0])+1.), color='skyblue')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,63]) / (np.arange(z_samples.shape[0])+1.), color='black')
    ax[1].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,87]) / (np.arange(z_samples.shape[0])+1.), color='#009E73')
    ax[1].set_xlim([1,z_samples.shape[0]])
    ax[1].set_ylim([0,1])
    ax[1].set_ylabel('Mean values')
    ax[2].hist(lam_samples, 50)
    ax[2].set_xlabel('Lambda values')
    ax[2].set_ylabel('Lambda')
    ax[3].scatter(x, probs, alpha=0.5)
    ax[3].plot(x, z, lw=2, color='orange')
    ax[3].fill_between(x, z_lower, z_upper, alpha=0.3, color='orange')
    ax[3].set_ylim([0,1])
    ax[3].set_xlim([0,1])
    ax[3].set_ylabel('Probability of success')
    plt.show()
    plt.clf()

def test_sample_gtf_gaussian():
    # Load the data and create the penalty matrix
    k = 0
    y = np.zeros(100)
    y[:25] = 15.
    y[25:50] = 20.
    y[50:75] = 25.
    y[75:] = 10.
    # y = (np.sin(np.linspace(-np.pi, np.pi, 100)) + 1) * 5
    # y[25:75] += np.sin(np.linspace(1.5*-np.pi, np.pi*2, 50))*5 ** (np.abs(np.arange(50) / 25.))
    # firsty = int(np.floor(len(y)/2))
    # secondy = int(np.ceil(len(y)/2))
    #y += np.concatenate((np.random.normal(0,0.1,size=firsty), np.random.normal(0,4.0,size=secondy)))
    y += np.random.normal(0,1,size=len(y))
    mean_offset = y.mean()
    y -= mean_offset
    stdev_offset = y.std()
    y /= stdev_offset
    
    # equally weight each data point
    w = np.ones(len(y))

    # try different weights for each data point
    # w = np.ones(len(y))
    # w[0:len(y)/2] = 100
    # w[len(y)/2:] = 10
    
    D = get_1d_penalty_matrix(len(y))

    z_samples, lam_samples = sample_gtf((y, w), D, k, likelihood='gaussian', prior='laplacegamma', robust=True, verbose=True, iterations=10000, burn=1000, thin=2)
    y *= stdev_offset
    y += mean_offset
    z_samples *= stdev_offset
    z_samples += mean_offset
    z = z_samples.mean(axis=0)
    z_stdev = z_samples.std(axis=0)
    z_lower = z - z_stdev*2
    z_upper = z + z_stdev*2
    assert(len(z) == len(y))


    fig, ax = plt.subplots(3)
    x = np.linspace(0,1,len(y))
    ax[0].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,12]) / (np.arange(z_samples.shape[0])+1.), color='orange')
    ax[0].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,37]) / (np.arange(z_samples.shape[0])+1.), color='skyblue')
    ax[0].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,63]) / (np.arange(z_samples.shape[0])+1.), color='black')
    ax[0].plot(np.arange(z_samples.shape[0])+1, np.cumsum(z_samples[:,87]) / (np.arange(z_samples.shape[0])+1.), color='#009E73')
    ax[0].set_xlim([1,z_samples.shape[0]])
    ax[0].set_ylabel('Sample beta mean values')

    n, bins, patches = ax[1].hist(lam_samples, 50)
    ax[1].set_xlabel('Lambda values')
    ax[1].set_ylabel('Lambda')

    ax[2].scatter(x, y, alpha=0.5)
    ax[2].plot(x, z, lw=2, color='orange')
    ax[2].fill_between(x, z_lower, z_upper, alpha=0.3, color='orange')
    ax[2].set_xlim([0,1])
    ax[2].set_ylabel('y')
    
    plt.show()
    plt.clf()

if __name__ == '__main__':
    test_sample_gtf_gaussian()