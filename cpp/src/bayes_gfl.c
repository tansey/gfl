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
 
 #include "bayes_gfl.h"

void bayes_gfl_gaussian_laplace (int n, double *y, double *w,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda_hyperparam_a, double lambda_hyperparam_b,
                                int iterations, int burn, int thin,
                                double **beta_samples, double *lambda_samples)
{
    int i;
    double *s;
    double *beta;
    int **coefs;
    int *coef_breaks;
    int iteration;
    int sample_idx;
    double ymean;
    const gsl_rng_type *T;
    gsl_rng *random;
    double lambda;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    random = gsl_rng_alloc (T);

    s = (double *) malloc(dk_rows * sizeof(double));
    coefs = (int **) malloc(n * sizeof(int*));
    coef_breaks = (int *) malloc(n * sizeof(int));
    beta = (double *) malloc(n * sizeof(double));

    /* Cache a lookup table to map from deltak column to the set of rows with 
       non-zero entries for that column */
    calc_coefs(n, dk_rows, dk_rowbreaks, dk_cols, coefs, coef_breaks);

    /* Set all beta values to the mean to start */
    ymean = vec_mean(n, y);
    for (i = 0; i < n; i++){ beta[i] = ymean; }

    /* Run the Gibbs sampler */
    for (iteration = 0, sample_idx = 0; iteration < iterations; iteration++)
    {
        /* Sample the lambda penalty weight on the Laplace prior */
        lambda = sample_lambda_laplace(random, beta,
                                        dk_rows, dk_rowbreaks, dk_cols, deltak,
                                        lambda_hyperparam_a, lambda_hyperparam_b);

        /* Sample each of the auxillary variables (one per row of Dk) */
        sample_prior_aux_laplace(random, beta, dk_rows, dk_rowbreaks, dk_cols, deltak, lambda, s);

        /* Sample from the truncated Gaussian likelihood */
        sample_likelihood_gaussian(random, n, y, w, dk_rowbreaks, dk_cols, deltak, s, coefs, coef_breaks, beta);

        /* Add the sample */
        if (iteration >= burn && (iteration % thin) == 0){
            lambda_samples[sample_idx] = lambda;
            memcpy(beta_samples[sample_idx], beta, n * sizeof(double));
            sample_idx++;
        }
    }

    free(s);
    free(beta);
    for (i = 0; i < n; i++){ free(coefs[i]); }
    free(coefs);
    free(coef_breaks);

    gsl_rng_free(random);
}


void bayes_gfl_gaussian_doublepareto (int n, double *y, double *w,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double lam_walk_stdev, double lam0, double dp_hyperparameter,
                                      int iterations, int burn, int thin,
                                      double **beta_samples, double *lambda_samples)
{
    int i;
    double *s;
    double *beta;
    int **coefs;
    int *coef_breaks;
    int iteration;
    int sample_idx;
    double ymean;
    const gsl_rng_type *T;
    gsl_rng *random;
    double lambda;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    random = gsl_rng_alloc (T);

    s = (double *) malloc(dk_rows * sizeof(double));
    coefs = (int **) malloc(n * sizeof(int*));
    coef_breaks = (int *) malloc(n * sizeof(int));
    beta = (double *) malloc(n * sizeof(double));
    lambda = lam0;

    /* Cache a lookup table to map from deltak column to the set of rows with 
       non-zero entries for that column */
    calc_coefs(n, dk_rows, dk_rowbreaks, dk_cols, coefs, coef_breaks);

    /* Set all beta values to the mean to start */
    ymean = vec_mean(n, y);
    for (i = 0; i < n; i++){ beta[i] = ymean; }

    /* Run the Gibbs sampler */
    for (iteration = 0, sample_idx = 0; iteration < iterations; iteration++)
    {
        /* Sample the lambda penalty weight on the double Pareto prior */
        lambda = sample_lambda_doublepareto(random, beta,
                                            dk_rows, dk_rowbreaks, dk_cols, deltak,
                                            lambda_hyperparam_a, lambda_hyperparam_b,
                                            lambda, dp_hyperparameter, lam_walk_stdev);

        /* Sample each of the auxillary variables (one per row of Dk) */
        sample_prior_aux_doublepareto(random, beta, dk_rows, dk_rowbreaks, dk_cols, deltak, lambda, dp_hyperparameter, s);
        
        /* Sample from the truncated Gaussian likelihood */
        sample_likelihood_gaussian(random, n, y, w, dk_rowbreaks, dk_cols, deltak, s, coefs, coef_breaks, beta);

        /* Add the sample */
        if (iteration >= burn && (iteration % thin) == 0){
            lambda_samples[sample_idx] = lambda;
            memcpy(beta_samples[sample_idx], beta, n * sizeof(double));
            sample_idx++;
        }
    }

    free(s);
    free(beta);
    for (i = 0; i < n; i++){ free(coefs[i]); }
    free(coefs);
    free(coef_breaks);

    gsl_rng_free(random);
}

void bayes_gfl_binomial_laplace (int n, int *trials, int *successes,
                                 int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                 double lambda_hyperparam_a, double lambda_hyperparam_b,
                                 int iterations, int burn, int thin,
                                 double **beta_samples, double *lambda_samples)
{
    int i;
    double *s;
    double *beta;
    int **coefs;
    int *coef_breaks;
    int iteration;
    int sample_idx;
    double ymean;
    const gsl_rng_type *T;
    gsl_rng *random;
    double lambda;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    random = gsl_rng_alloc (T);

    s = (double *) malloc(dk_rows * sizeof(double));
    coefs = (int **) malloc(n * sizeof(int*));
    coef_breaks = (int *) malloc(n * sizeof(int));
    beta = (double *) malloc(n * sizeof(double));

    /* Cache a lookup table to map from deltak column to the set of rows with 
       non-zero entries for that column */
    calc_coefs(n, dk_rows, dk_rowbreaks, dk_cols, coefs, coef_breaks);

    /* Set all beta values to the mean to start */
    ymean = 0;
    for (i = 0; i < n; i++){ if (successes[i] > 0) {ymean += trials[i] / (double)successes[i];} }
    ymean = -gsl_sf_log(ymean / (double)n);
    for (i = 0; i < n; i++){ beta[i] = ymean; }

    /* Run the Gibbs sampler */
    for (iteration = 0, sample_idx = 0; iteration < iterations; iteration++)
    {
        /* Sample the lambda penalty weight on the Laplace prior */
        lambda = sample_lambda_laplace(random, beta,
                                        dk_rows, dk_rowbreaks, dk_cols, deltak,
                                        lambda_hyperparam_a, lambda_hyperparam_b);

        
        /* Sample each of the auxillary variables (one per row of Dk) */
        sample_prior_aux_laplace(random, beta, dk_rows, dk_rowbreaks, dk_cols, deltak, lambda, s);
        
        /* Sample from the binomial likelihood */
        sample_likelihood_binomial(random, n, trials, successes, dk_rowbreaks, dk_cols, deltak, s, coefs, coef_breaks, beta);
        
        /* Add the sample */
        if (iteration >= burn && (iteration % thin) == 0){
            lambda_samples[sample_idx] = lambda;
            for(i = 0; i < n; i++){ beta_samples[sample_idx][i] = beta[i] < -200 ? 0 : 1.0 / (1.0 + gsl_sf_exp(-beta[i])); }
            sample_idx++;
        }
    }

    free(s);
    free(beta);
    for (i = 0; i < n; i++){ free(coefs[i]); }
    free(coefs);
    free(coef_breaks);

    gsl_rng_free(random);
}

void bayes_gfl_binomial_doublepareto (int n, int *trials, int *successes,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double lam_walk_stdev, double lam0, double dp_hyperparameter,
                                      int iterations, int burn, int thin,
                                      double **beta_samples, double *lambda_samples)
{
    int i;
    double *s;
    double *beta;
    int **coefs;
    int *coef_breaks;
    int iteration;
    int sample_idx;
    double ymean;
    const gsl_rng_type *T;
    gsl_rng *random;
    double lambda;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    random = gsl_rng_alloc (T);
    lambda = lam0;

    s = (double *) malloc(dk_rows * sizeof(double));
    coefs = (int **) malloc(n * sizeof(int*));
    coef_breaks = (int *) malloc(n * sizeof(int));
    beta = (double *) malloc(n * sizeof(double));

    /* Cache a lookup table to map from deltak column to the set of rows with 
       non-zero entries for that column */
    calc_coefs(n, dk_rows, dk_rowbreaks, dk_cols, coefs, coef_breaks);

    /* Set all beta values to the mean to start */
    ymean = 0;
    for (i = 0; i < n; i++){ if (successes[i] > 0) {ymean += trials[i] / (double)successes[i];} }
    ymean = -gsl_sf_log(ymean / (double)n);
    for (i = 0; i < n; i++){ beta[i] = ymean; }

    /* Run the Gibbs sampler */
    for (iteration = 0, sample_idx = 0; iteration < iterations; iteration++)
    {
        /* Sample the lambda penalty weight on the double Pareto prior */
        lambda = sample_lambda_doublepareto(random, beta,
                                            dk_rows, dk_rowbreaks, dk_cols, deltak,
                                            lambda_hyperparam_a, lambda_hyperparam_b,
                                            lambda, dp_hyperparameter, lam_walk_stdev);

        /* Sample each of the auxillary variables (one per row of Dk) */
        sample_prior_aux_doublepareto(random, beta, dk_rows, dk_rowbreaks, dk_cols, deltak, lambda, dp_hyperparameter, s);

        /* Sample from the binomial likelihood */
        sample_likelihood_binomial(random, n, trials, successes, dk_rowbreaks, dk_cols, deltak, s, coefs, coef_breaks, beta);

        /* Add the sample */
        if (iteration >= burn && (iteration % thin) == 0){
            lambda_samples[sample_idx] = lambda;
            for(i = 0; i < n; i++){ beta_samples[sample_idx][i] = beta[i] < -200 ? 0 : 1.0 / (1.0 + gsl_sf_exp(-beta[i])); }
            sample_idx++;
        }
    }

    free(s);
    free(beta);
    for (i = 0; i < n; i++){ free(coefs[i]); }
    free(coefs);
    free(coef_breaks);

    gsl_rng_free(random);
}

void bayes_gfl_poisson_laplace (int n, int *obs,
                                 int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                 double lambda_hyperparam_a, double lambda_hyperparam_b,
                                 int iterations, int burn, int thin,
                                 double **beta_samples, double *lambda_samples)
{
    int i;
    double *s;
    double *beta;
    int **coefs;
    int *coef_breaks;
    int iteration;
    int sample_idx;
    double ymean;
    const gsl_rng_type *T;
    gsl_rng *random;
    double lambda;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    random = gsl_rng_alloc (T);

    s = (double *) malloc(dk_rows * sizeof(double));
    coefs = (int **) malloc(n * sizeof(int*));
    coef_breaks = (int *) malloc(n * sizeof(int));
    beta = (double *) malloc(n * sizeof(double));

    /* Cache a lookup table to map from deltak column to the set of rows with 
       non-zero entries for that column */
    calc_coefs(n, dk_rows, dk_rowbreaks, dk_cols, coefs, coef_breaks);

    /* Set all beta values to the mean to start */
    ymean = gsl_sf_log(vec_mean_int(n, obs));
    for (i = 0; i < n; i++){ beta[i] = ymean; }

    /* Run the Gibbs sampler */
    for (iteration = 0, sample_idx = 0; iteration < iterations; iteration++)
    {
        /* Sample the lambda penalty weight on the Laplace prior */
        lambda = sample_lambda_laplace(random, beta,
                                        dk_rows, dk_rowbreaks, dk_cols, deltak,
                                        lambda_hyperparam_a, lambda_hyperparam_b);

        
        /* Sample each of the auxillary variables (one per row of Dk) */
        sample_prior_aux_laplace(random, beta, dk_rows, dk_rowbreaks, dk_cols, deltak, lambda, s);
        
        /* Sample from the Poisson likelihood */
        sample_likelihood_poisson(random, n, obs, dk_rowbreaks, dk_cols, deltak, s, coefs, coef_breaks, beta);
        
        /* Add the sample */
        if (iteration >= burn && (iteration % thin) == 0){
            lambda_samples[sample_idx] = lambda;
            for(i = 0; i < n; i++){ beta_samples[sample_idx][i] = gsl_sf_exp(beta[i]); }
            sample_idx++;
        }
    }

    free(s);
    free(beta);
    for (i = 0; i < n; i++){ free(coefs[i]); }
    free(coefs);
    free(coef_breaks);

    gsl_rng_free(random);
}

void bayes_gfl_poisson_doublepareto (int n, int *obs,
                                      int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                      double lambda_hyperparam_a, double lambda_hyperparam_b,
                                      double lam_walk_stdev, double lam0, double dp_hyperparameter,
                                      int iterations, int burn, int thin,
                                      double **beta_samples, double *lambda_samples)
{
    int i;
    double *s;
    double *beta;
    int **coefs;
    int *coef_breaks;
    int iteration;
    int sample_idx;
    double ymean;
    const gsl_rng_type *T;
    gsl_rng *random;
    double lambda;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    random = gsl_rng_alloc (T);

    s = (double *) malloc(dk_rows * sizeof(double));
    coefs = (int **) malloc(n * sizeof(int*));
    coef_breaks = (int *) malloc(n * sizeof(int));
    beta = (double *) malloc(n * sizeof(double));
    lambda = lam0;

    /* Cache a lookup table to map from deltak column to the set of rows with 
       non-zero entries for that column */
    calc_coefs(n, dk_rows, dk_rowbreaks, dk_cols, coefs, coef_breaks);

    /* Set all beta values to the mean to start */
    ymean = gsl_sf_log(vec_mean_int(n, obs));
    for (i = 0; i < n; i++){ beta[i] = ymean; }

    /* Run the Gibbs sampler */
    for (iteration = 0, sample_idx = 0; iteration < iterations; iteration++)
    {
        /* Sample the lambda penalty weight on the double Pareto prior */
        lambda = sample_lambda_doublepareto(random, beta,
                                            dk_rows, dk_rowbreaks, dk_cols, deltak,
                                            lambda_hyperparam_a, lambda_hyperparam_b,
                                            lambda, dp_hyperparameter, lam_walk_stdev);

        /* Sample each of the auxillary variables (one per row of Dk) */
        sample_prior_aux_doublepareto(random, beta, dk_rows, dk_rowbreaks, dk_cols, deltak, lambda, dp_hyperparameter, s);
        
        /* Sample from the Poisson likelihood */
        sample_likelihood_poisson(random, n, obs, dk_rowbreaks, dk_cols, deltak, s, coefs, coef_breaks, beta);
        
        /* Add the sample */
        if (iteration >= burn && (iteration % thin) == 0){
            lambda_samples[sample_idx] = lambda;
            for(i = 0; i < n; i++){ beta_samples[sample_idx][i] = gsl_sf_exp(beta[i]); }
            sample_idx++;
        }
    }

    free(s);
    free(beta);
    for (i = 0; i < n; i++){ free(coefs[i]); }
    free(coefs);
    free(coef_breaks);

    gsl_rng_free(random);
}


double sample_lambda_laplace(const gsl_rng *random, double *beta, 
                               int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                               double a, double b)
{
    double *x;
    double lambda;

    x = (double *) malloc(dk_rows * sizeof(double));
    mat_dot_vec(dk_rows, dk_rowbreaks, dk_cols, deltak, beta, x);
    vec_abs(dk_rows, x);

    lambda = gsl_ran_gamma(random, a+dk_rows, 1.0 / (b + vec_sum(dk_rows, x)));

    free(x);

    return lambda;
}

double sample_lambda_doublepareto(const gsl_rng *random, double *beta,
                                  int dk_rows, int *dk_rowbreaks, int *dk_cols, double *dk_vals,
                                  double a, double b,
                                  double lam0, double gamma, double lam_walk_stdev)
{
    int i;
    double lam1;
    double sum_term;
    double dotprod;
    double accept_ratio;
    int prev_break;

    lam1 = gsl_sf_exp(gsl_ran_gaussian(random, lam_walk_stdev) + gsl_sf_log(lam0));

    sum_term = 0;
    prev_break = 0;
    for(i = 0; i < dk_rows; i++){
        dotprod = fabs(vec_dot_beta(dk_rowbreaks[i] - prev_break, dk_cols + prev_break, dk_vals + prev_break, beta));
        sum_term += gsl_sf_log(1 + dotprod / (gamma * lam1)) - gsl_sf_log(1 + dotprod / (gamma * lam0));
        prev_break = dk_rowbreaks[i];
    }

    accept_ratio = gsl_sf_exp((a - 1 - dk_rows) * (gsl_sf_log(lam1) - gsl_sf_log(lam0)) - b * (lam1 - lam0) - (gamma + 1) * sum_term);
    if (accept_ratio >= 1 || gsl_ran_flat(random, 0, 1) <= accept_ratio)
        return lam1;
    return lam0;
}

void sample_prior_aux_laplace(const gsl_rng *random, double *beta,
                                int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                double lambda, double *s)
{
    int i;

    mat_dot_vec(dk_rows, dk_rowbreaks, dk_cols, deltak, beta, s);
    vec_abs(dk_rows, s);

    for(i = 0; i < dk_rows; i++){
        s[i] = -gsl_sf_log(gsl_ran_flat (random, 0.0, gsl_sf_exp(-lambda * s[i]))) / lambda;
    }
}

void sample_prior_aux_doublepareto(const gsl_rng *random, double *beta, 
                                   int dk_rows, int *dk_rowbreaks, int *dk_cols, double *deltak,
                                   double lambda, double dp_hyperparameter, double *s)
{
    int i;
    double z;

    mat_dot_vec(dk_rows, dk_rowbreaks, dk_cols, deltak, beta, s);
    vec_abs(dk_rows, s);

    for(i = 0; i < dk_rows; i++){
        z = pow(1. + s[i] / (dp_hyperparameter * lambda), -dp_hyperparameter - 1.);
        s[i] = dp_hyperparameter * lambda * (gsl_sf_exp(-gsl_sf_log(gsl_ran_flat(random, 0, z)) / (dp_hyperparameter + 1.0)) - 1);
    }
}

void sample_likelihood_gaussian(const gsl_rng *random,
                                int n, double *y, double *w,
                                int *dk_rowbreaks, int *dk_cols, double *dk_vals,
                                double *s, int **coefs, int *coef_breaks,
                                double *beta)
{
    int i;
    int j;
    int k;
    int row;
    int j_idx;
    double a;
    double lower;
    double upper;
    double left;
    double right;
    
    for(j = 0; j < n; j++)
    {
        lower = -INFINITY;
        upper = INFINITY;
        /* Bound the sampling range */
        for (i = 0; i < coef_breaks[j]; i++)
        {
            /* current row that has a non-zero value for column j */
            row = coefs[j][i];

            /* Calculate Dk[i].dot(b_notj), the inner product of the i'th row
               and the beta vector, excluding the j'th column. */
            a = 0;
            for (k = row == 0 ? 0 : dk_rowbreaks[row-1]; k < dk_rowbreaks[row]; k++){
                if (dk_cols[k] == j){
                    j_idx = k;
                } else{
                    a += dk_vals[k] * beta[dk_cols[k]];
                }
            }

            /* Find the left and right bounds */
            left = (-s[row] - a) / dk_vals[j_idx];
            right = (s[row] - a) / dk_vals[j_idx];
            if (dk_vals[j_idx] >= 0){
                lower = MAX(lower, left);
                upper = MIN(upper, right);
            } else {
                lower = MAX(lower, right);
                upper = MIN(upper, left);
            }

            beta[j] = rnorm_trunc(random, y[j], 1. / sqrt(w[j]), lower, upper);
        }
    }

}

void sample_likelihood_binomial(const gsl_rng *random,
                                int n, int *trials, int *successes, 
                                int *dk_rowbreaks, int *dk_cols, double *dk_vals,
                                double *s, int **coefs, int *coef_breaks,
                                double *beta)
{
    int i;
    int j;
    int k;
    int row;
    int j_idx;
    double a;
    double lower;
    double upper;
    double left;
    double right;
    
    for(j = 0; j < n; j++)
    {
        if (successes[j] > 0){
            lower = gsl_ran_flat(random, 0, gsl_sf_exp(successes[j]*beta[j]));
            lower = gsl_sf_log(lower) / (double)successes[j];
        } else {
            lower = -INFINITY;
        }
        upper = gsl_ran_flat(random, 0, beta[j] < -160 ? 1 : gsl_sf_exp(-trials[j] * gsl_sf_log(1 + gsl_sf_exp(beta[j]))));
        upper = gsl_sf_log(pow(upper,-1.0/(double)trials[j]) - 1);
        /* Bound the sampling range */
        for (i = 0; i < coef_breaks[j]; i++)
        {
            /* current row that has a non-zero value for column j */
            row = coefs[j][i];
            
            /* Calculate Dk[i].dot(b_notj), the inner product of the i'th row
               and the beta vector, excluding the j'th column. */
            a = 0;
            for (k = row == 0 ? 0 : dk_rowbreaks[row-1]; k < dk_rowbreaks[row]; k++){
                if (dk_cols[k] == j){
                    j_idx = k;
                } else{
                    a += dk_vals[k] * beta[dk_cols[k]];
                }
            }
            
            /* Find the left and right bounds */
            left = (-s[row] - a) / dk_vals[j_idx];
            right = (s[row] - a) / dk_vals[j_idx];
            if (dk_vals[j_idx] >= 0){
                lower = MAX(lower, left);
                upper = MIN(upper, right);
            } else {
                lower = MAX(lower, right);
                upper = MIN(upper, left);
            }
            
            beta[j] = gsl_ran_flat(random, lower, upper);
        }
    }
}

void sample_likelihood_poisson(const gsl_rng *random,
                                int n, int *obs, 
                                int *dk_rowbreaks, int *dk_cols, double *dk_vals,
                                double *s, int **coefs, int *coef_breaks,
                                double *beta)
{
    int i;
    int j;
    int k;
    int row;
    int j_idx;
    double a;
    double lower;
    double upper;
    double left;
    double right;
    
    for(j = 0; j < n; j++)
    {
        if (obs[j] > 0){
            lower = gsl_ran_flat(random, 0, gsl_sf_exp(obs[j]*beta[j]));
            lower = gsl_sf_log(lower) / (double)obs[j];
        } else {
            lower = -INFINITY;
        }
        upper = gsl_ran_flat(random, 0, beta[j] < -160 ? 1 : gsl_sf_exp(-gsl_sf_exp(beta[j])));
        upper = gsl_sf_log(-gsl_sf_log(upper));
        /* Bound the sampling range */
        for (i = 0; i < coef_breaks[j]; i++)
        {
            /* current row that has a non-zero value for column j */
            row = coefs[j][i];
            
            /* Calculate Dk[i].dot(b_notj), the inner product of the i'th row
               and the beta vector, excluding the j'th column. */
            a = 0;
            for (k = row == 0 ? 0 : dk_rowbreaks[row-1]; k < dk_rowbreaks[row]; k++){
                if (dk_cols[k] == j){
                    j_idx = k;
                } else{
                    a += dk_vals[k] * beta[dk_cols[k]];
                }
            }
            
            /* Find the left and right bounds */
            left = (-s[row] - a) / dk_vals[j_idx];
            right = (s[row] - a) / dk_vals[j_idx];
            if (dk_vals[j_idx] >= 0){
                lower = MAX(lower, left);
                upper = MIN(upper, right);
            } else {
                lower = MAX(lower, right);
                upper = MIN(upper, left);
            }
            
            beta[j] = gsl_ran_flat(random, lower, upper);
        }
    }
}

void calc_coefs(int n, 
                int dk_rows, int *dk_rowbreaks, int *dk_cols,
                int **coefs, int *coef_breaks)
{
    int i;
    int row;
    int *counts;

    counts = (int *) malloc(n * sizeof(int));

    /* Zero-out the counts array */
    for (i = 0; i < n; i++){ counts[i] = 0; }

    /* First pass -- figure out the length of each coef array */
    i = 0;
    for (row = 0; row < dk_rows; row++){
        for (; i < dk_rowbreaks[row]; i++){
            counts[dk_cols[i]]++;
        }
    }

    /* Create each coef array */
    coefs[0] = (int *) malloc(counts[0] * sizeof(int));
    coef_breaks[0] = counts[0];
    for (i = 1; i < n; i++){
        coefs[i] = (int *) malloc(counts[i] * sizeof(int));
        coef_breaks[i] = counts[i];
    }

    /* Second pass -- fill in coefs */
    i = 0;
    for (row = 0; row < dk_rows; row++){
        for (; i < dk_rowbreaks[row]; i++){
            coefs[dk_cols[i]][counts[dk_cols[i]] - 1] = row;
            counts[dk_cols[i]]--;
        }
    }

    free(counts);
}



