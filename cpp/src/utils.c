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
 
 #include "utils.h"

void print_vector(int n, double *v)
{
    int i;
    printf("[");
    for (i = 0; i < n; i++)
        printf(" %f", v[i]);
    printf("]");
}

double lex_ran_flat(const gsl_rng *random, double lower, double upper)
{
    double u;
    if (lower == upper) { return upper; }
    do { u = gsl_ran_flat(random, lower, upper); } while (u == lower);
    return u;
}

double vec_mean(int n, double *x)
{
    int i;
    double mean;

    mean = 0;

    for (i = 0; i < n; i++) { mean += x[i] / (double)n; }

    return mean;
}

double vec_mean_int(int n, int *x)
{
    int i;
    double mean;

    mean = 0;

    for (i = 0; i < n; i++) { mean += x[i] / (double)n; }

    return mean;
}

double vec_norm(int n, double *x)
{
    int i;
    double r;
    for(r = 0, i = 0; i < n; i++){ r += x[i] * x[i]; }
    return sqrt(r);
}

void mat_dot_vec(int nrows, int *rowbreaks, int *cols, double *A, double *x, double *b)
{
    int i;
    int row;

    i = 0;
    for (row = 0; row < nrows; row++){
        b[row] = 0;
        for (; i < rowbreaks[row]; i++){
            b[row] += A[i] * x[cols[i]];
        }
    }
}

void vec_abs(int n, double *x)
{
    int i;

    for(i = 0; i < n; i++){
        x[i] = x[i] < 0 ? -x[i] : x[i];
    }
}

double vec_sum(int n, double *x)
{
    int i;
    double sum;

    sum = 0;
    for(i = 0; i < n; i++){
        sum += x[i];
    }
    return sum;
}

double vec_dot_beta(int n, int *cols, double *vals, double *beta)
{
    int i;
    double sum_term;

    sum_term = 0;
    for (i = 0; i < n; i++){
        sum_term += vals[i] * beta[cols[i]];
    }
    return sum_term;
}

void vec_plus_vec(int n, double *a, double *b, double *c){
    int i;
    for (i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}
void vec_minus_vec(int n, double *a, double *b, double *c){
    int i;
    for (i = 0; i < n; i++){
        c[i] = a[i] - b[i];
    }
}

double vec_dot_vec(int n, double *a, double *b){
    int i;
    double r;
    for (i = 0, r = 0; i < n; i++){
        r += a[i] * b[i];
    }
    return r;
}

/*//////////
// Utilities for drawing truncated normals
// just found an Rcpp version of piecewise rejection sampling on stackoverflow and translated it to C with GSL:
//    http://stackoverflow.com/questions/17915234/fast-sampling-from-truncated-normal-distribution-using-rcpp-and-openmp
// It is probably not optimal but I just needed something off the shelf.
//////////*/

/* norm_rs(a, b)
// generates a sample from a N(0,1) RV restricted to be in the interval
// (a,b) via rejection sampling.
// ====================================================================== */
double norm_rs(const gsl_rng *random, double a, double b)
{
    double  x;
    x = gsl_ran_ugaussian(random);
    while( (x < a) || (x > b) ) {
        x = gsl_ran_ugaussian(random);
    }
    return x;
}

/* exp_rs(a, b)
// generates a sample from a N(0,1) RV restricted to the interval
// (a,b) using exponential rejection sampling.
// ====================================================================== */
double exp_rs(const gsl_rng *random, double a, double b)
{
    double z;
    double u;
    double rate;

    rate = 1.0 / a;
    /* Generate a proposal on (0, b-a) */
    z = gsl_ran_exponential(random, rate);
    while(z > (b-a)){
        z = gsl_ran_exponential(random, rate);
    }
    u = lex_ran_flat(random, 0.0, 1.0);

    while( gsl_sf_log(u) > (-0.5*z*z))
    {
        z = gsl_ran_exponential(random, rate);
        while(z > (b-a)){
            z = gsl_ran_exponential(random, rate);  
        }
        u = lex_ran_flat(random, 0.0, 1.0);
    }
    return (z+a);
}

/* half_norm_rs(a, b)
// generates a sample from a N(0,1) RV restricted to the interval
// (a,b) (with a > 0) using half normal rejection sampling.
// ====================================================================== */
double half_norm_rs(const gsl_rng *random, double a, double b)
{
    double x;
    x = fabs(gsl_ran_ugaussian(random));
    while( (x<a) || (x>b) ) {
        x = fabs(gsl_ran_ugaussian(random));
    }
    return x;
}

/* unif_rs(a, b)
// generates a sample from a N(0,1) RV restricted to the interval
// (a,b) using uniform rejection sampling. 
// ====================================================================== */
double unif_rs(const gsl_rng *random, double a, double b)
{
   double xstar;
   double logphixstar;
   double x;
   double u;
   double logu;

   /* Find the argmax (b is always >= 0)
      This works because we want to sample from N(0,1) */
    if(a <= 0.0) {
        xstar = 0.0;
    } else {
        xstar = a;
    }
    logphixstar = log_norm_pdf(xstar, 0, 1);
    x = gsl_ran_flat(random, a, b);
    u = lex_ran_flat(random, 0.0, 1.0);
    logu = gsl_sf_log(u);
    while( logu > (log_norm_pdf(x, 0, 1) - logphixstar))
    {
        x = gsl_ran_flat(random, a, b);
        u = lex_ran_flat(random, 0.0, 1.0);
        logu = gsl_sf_log(u);
    }
    return x;
}


/* rnorm_trunc( mu, sigma, lower, upper)
//
// generates one random normal RV with mean 'mu' and standard
// deviation 'sigma', truncated to the interval (lower,upper), where
// lower can be -Inf and upper can be Inf.
// ====================================================================== */
double rnorm_trunc (const gsl_rng *random, double mu, double sigma, double lower, double upper)
{
    int change;
    double a;
    double b;
    double logt1;
    double logt2;
    double t3;
    double z;
    double tmp;
    double lograt;

    logt1 = gsl_sf_log(0.150);
    logt2 = gsl_sf_log(2.18);
    t3 = 0.725;

    change = 0;
    a = (lower - mu)/sigma;
    b = (upper - mu)/sigma;

    /* First scenario */
    if( (a == -INFINITY) || (b == INFINITY))
    {
        if(a == -INFINITY) {
            change = 1;
            a = -b;
            b = INFINITY;
        }
        /* The two possibilities for this scenario */
        if(a <= 0.45) { z = norm_rs(random, a, b); }
        else { z = exp_rs(random, a, b); }
        if(change) { z = -z; }
    }
    /* Second scenario */
    else if((a * b) <= 0.0)
    {
        /* The two possibilities for this scenario */
        if((log_norm_pdf(a, 0, 1) <= logt1) || (log_norm_pdf(b, 0, 1) <= logt1))
        {
            z = norm_rs(random, a, b);
        }
        else { z = unif_rs(random, a,b); }
    }
    /* Third scenario */
    else
    {
        if(b < 0) {
            tmp = b;
            b = -a;
            a = -tmp;
            change = 1;
        }
        lograt = log_norm_pdf(a, 0, 1) - log_norm_pdf(b, 0, 1);
        if(lograt <= logt2) { z = unif_rs(random, a,b); }
        else if((lograt > logt1) && (a < t3)) { z = half_norm_rs(random, a,b); }
        else { z = exp_rs(random, a,b); }
        if(change) { z = -z; }
    }
    
    return (sigma*z + mu);
}

/* Runs the truncated normal function without having to pass a random number generator */
double rnorm_trunc_norand (double mu, double sigma, double lower, double upper)
{
    const gsl_rng_type *T;
    gsl_rng *random;
    double x;
    long seed;

    T = gsl_rng_default;
    random = gsl_rng_alloc (T);
    seed = time (NULL) * clock();    
    gsl_rng_set (random, seed);
    
    x = rnorm_trunc(random, mu, sigma, lower, upper);
    
    gsl_rng_free(random);

    return x;
}

double log_norm_pdf(double x, double mu, double sigma)
{
    x = (x - mu) / sigma;
    return -(M_LN_SQRT_2PI + 0.5 * x * x + gsl_sf_log(sigma));
}



/*//////////
// Utilities for doing sparse matrix and dense vector products when you have
// an output vector.
//////////*/
void cs_dot_vec(cs *A, double *x, double *b){
    int i;
    int j;

    for (i = 0; i < A->m; i++){ b[i] = 0; } /* zero-out the output vector */

    if (A->nz == -1){
        for (i = 0, j = 0; i < A->n; i++){
            for (; j < A->p[i+1]; j++){
                b[A->i[j]] += A->x[j] * x[i];
            }
        }
    } else {
        for (i = 0; i < A->nz; i++){
            b[A->i[i]] += A->x[i] * x[A->p[i]];
        }
    }
}

void vec_dot_cs(double *x, cs *A, double *b){
    int i;
    int j;

    for (i = 0; i < A->n; i++){ b[i] = 0; } /* zero-out the output vector */

    if (A->nz == -1){
        for (i = 0, j = 0; i < A->n; i++){
            for (; j < A->p[i+1]; j++){
                b[i] += A->x[j] * x[A->i[j]];
            }
        }
    } else {
        for (i = 0; i < A->nz; i++){
            b[A->p[i]] += A->x[i] * x[A->i[i]];
        }
    }
}



