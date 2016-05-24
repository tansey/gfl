/****************************************************************************
 * Copyright (C) 2016 by Wesley Tansey                                      *
 *                                                                          *
 * This file is part of the the GFL library / package.                      *
 *                                                                          *
 *   GFL is free software: you can redistribute it and/or                   *
 *   modify it under the terms of the GNU Lesser General Public License     *
 *   as published by the Free Software Foundation, either version 3 of      *
 *   the License, or (at your option) any later version.                    *
 *                                                                          *
 *   GFL is distributed in the hope that it will be useful,                 *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 *   GNU Lesser General Public License for more details.                    *
 *                                                                          *
 *   You should have received a copy of the GNU Lesser General Public       *
 *   License along with GFL. If not, see <http://www.gnu.org/licenses/>.    *
 ****************************************************************************/

#ifndef BAYES_GFL_H
#define BAYES_GFL_H

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "polyagamma.h"
#include "utils.h"

#ifndef BINOMIAL_BETA_MIN
#define BINOMIAL_BETA_MIN -90
#endif
#ifndef BINOMIAL_BETA_MAX
#define BINOMIAL_BETA_MAX 90
#endif
#ifndef CLAMP
#define CLAMP(a,low,upper) MAX(low,MIN(upper,a))
#endif

/************************
 *     Main methods     *
 ************************/
void bayes_gfl_gaussian_laplace (int n, double *y, double *w,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda_hyperparam_a, double lambda_hyperparam_b,
                                long iterations, long burn, long thin,
                                double **beta_samples, double *lambda_samples);

void bayes_gfl_gaussian_laplace_gamma (int n, double *y, double *w,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda_hyperparam_a, double lambda_hyperparam_b,
                                double tau_hyperparameter,
                                long iterations, long burn, long thin,
                                double **beta_samples, double *lambda_samples);

void bayes_gfl_gaussian_laplace_gamma_robust (int n, double *y, double *w,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda_hyperparam_a, double lambda_hyperparam_b,
                                double tau_hyperparameter,
                                double w_hyperparameter_a, double w_hyperparameter_b,
                                long iterations, long burn, long thin,
                                double **beta_samples, double *lambda_samples);

void bayes_gfl_gaussian_doublepareto (int n, double *y, double *w,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double lam_walk_stdev, double lam0, double dp_hyperparameter,
                                      long iterations, long burn, long thin,
                                      double **beta_samples, double *lambda_samples);

/* mixture-of-laplaces representation */
void bayes_gfl_gaussian_doublepareto2 (int n, double *y, double *w,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double tau_hyperparameter,
                                      long iterations, long burn, long thin,
                                      double **beta_samples, double *lambda_samples);

void bayes_gfl_gaussian_cauchy (int n, double *y, double *w,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double lam_walk_stdev, double lam0,
                                      long iterations, long burn, long thin,
                                      double **beta_samples, double *lambda_samples);

void bayes_gfl_binomial_laplace (int n, int *trials, int *successes,
                                 int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                 double lambda_hyperparam_a, double lambda_hyperparam_b,
                                 long iterations, long burn, long thin,
                                 double **beta_samples, double *lambda_samples);

void bayes_gfl_binomial_laplace_gamma (int n, int *trials, int *successes,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda_hyperparam_a, double lambda_hyperparam_b,
                                double tau_hyperparameter,
                                long iterations, long burn, long thin,
                                double **beta_samples, double *lambda_samples);

void empirical_bayes_gfl_binomial_laplace_gamma (int n, int *trials, int *successes,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda,
                                long iterations, long burn, long thin,
                                double **beta_samples, double *lambda_samples);

void bayes_gfl_binomial_doublepareto (int n, int *trials, int *successes,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double lam_walk_stdev, double lam0, double dp_hyperparameter,
                                      long iterations, long burn, long thin,
                                      double **beta_samples, double *lambda_samples);

void bayes_gfl_poisson_laplace (int n, int *obs,
                                 int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                 double lambda_hyperparam_a, double lambda_hyperparam_b,
                                 long iterations, long burn, long thin,
                                 double **beta_samples, double *lambda_samples);

void bayes_gfl_poisson_doublepareto (int n, int *obs,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double lam_walk_stdev, double lam0, double dp_hyperparameter,
                                      long iterations, long burn, long thin,
                                      double **beta_samples, double *lambda_samples);

/********************
 *  Gibbs sampling  *
 ********************/
double sample_lambda_laplace(const gsl_rng *random, double *beta, 
                               int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                               double a, double b);

double sample_lambda_doublepareto(const gsl_rng *random, double *beta,
                                  int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                  double a, double b,
                                  double lam0, double gamma, double lam_walk_stdev);

double sample_lambda_doublepareto2(const gsl_rng *random, double *beta, 
                                   int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                   double a, double b, double gamma, double *tau);

double sample_lambda_cauchy(const gsl_rng *random, double *beta,
                                  int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                  double a, double b,
                                  double lam0, double lam_walk_stdev);

void sample_tau_laplace_gamma(const gsl_rng *random, double *beta,
                              int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                              double lambda, double tau_hyperparameter, 
                              double *tau);

void sample_prior_aux_laplace(const gsl_rng *random, double *beta,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda, double *s);

void sample_prior_aux_laplace_multilambda(const gsl_rng *random, double *beta,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double *lambda, double *s);

void sample_prior_aux_doublepareto(const gsl_rng *random, double *beta, 
                                   int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                   double lambda, double dp_hyperparameter, double *s);

void sample_prior_aux_cauchy(const gsl_rng *random, double *beta, 
                                   int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                   double lambda, double *s);

void sample_likelihood_gaussian(const gsl_rng *random,
                                int n, double *y, double *w,
                                int *dk_rowbreaks, int *dk_cols, double *dk_vals,
                                double *s, int **coefs, int *coef_breaks,
                                double *beta);

void sample_likelihood_binomial(const gsl_rng *random,
                                int n, int *trials, int *successes, 
                                int *dk_rowbreaks, int *dk_cols, double *dk_vals,
                                double *s, int **coefs, int *coef_breaks,
                                double *beta);

void sample_likelihood_poisson(const gsl_rng *random,
                                int n, int *obs, 
                                int *dk_rowbreaks, int *dk_cols, double *dk_vals,
                                double *s, int **coefs, int *coef_breaks,
                                double *beta);

/***********
 *  Utils  *
 ***********/
void calc_coefs(int n, 
                int dk_rows, int *dk_rowbreaks, int *dk_cols,
                int **coefs, int *coef_breaks);

#endif













