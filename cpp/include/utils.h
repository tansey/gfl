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

#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>
#include <time.h>
#include "csparse.h"

/* Define MAX and MIN functions */
#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef M_LN_SQRT_2PI
#define M_LN_SQRT_2PI   0.918938533204672741780329736406    /* log(sqrt(2*pi)) */
#endif


void print_vector(int n, double *v);
double lex_ran_flat(const gsl_rng *random, double lower, double upper); /* lower exclusive uniform random number */

double vec_mean(int n, double *x);
double vec_mean_int(int n, int *x);
void vec_abs(int n, double *x);
void vec_plus_vec(int n, double *a, double *b, double *c);
void vec_minus_vec(int n, double *a, double *b, double *c);
double vec_dot_vec(int n, double *a, double *b);
double vec_sum(int n, double *x);
double vec_dot_beta(int n, int *cols, double *vals, double *beta);
double vec_norm(int n, double *x);
void mat_dot_vec(int nrows, int *rowbreaks, int *cols, double *A, double *x, double *b);

/* Truncated normal sampling routines */
double norm_rs(const gsl_rng *random, double a, double b);
double exp_rs(const gsl_rng *random, double a, double b);
double half_norm_rs(const gsl_rng *random, double a, double b);
double unif_rs(const gsl_rng *random, double a, double b);
double rnorm_trunc (const gsl_rng *random, double mu, double sigma, double lower, double upper);
double rnorm_trunc_norand (double mu, double sigma, double lower, double upper);
double log_norm_pdf(double x, double mu, double sigma);

void cs_dot_vec(cs *A, double *x, double *b);
void vec_dot_cs(double *x, cs *A, double *b);

#endif

