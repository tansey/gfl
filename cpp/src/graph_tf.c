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
 
#include "graph_tf.h"

int graph_trend_filtering_weight_warm (int n, double *y, double *w, double lam,
                                       int dknrows, int dkncols, int dknnz,
                                       int *dkrows, int *dkcols, double *dkvals,
                                       int maxsteps, double rel_tol,
                                       double *beta, double *u)
{
    int i;
    int j;
    int step;
    int nfree;
    double cur_converge;
    double step_size;
    int *boundary;
    int *reduced_cols;
    double *gradient;
    double *reduced_gradient;
    double *reduced_direction;
    double *dky;
    double *udk;
    cs *dk;
    cs *wdkT;
    cs *hessian;
    cs *reduced_hessian;
    cs *temp;

    gradient                    = (double *) malloc(dknrows * sizeof(double));
    reduced_gradient            = (double *) malloc(dknrows * sizeof(double));
    reduced_direction           = (double *) malloc(dknrows * sizeof(double));
    dky                         = (double *) malloc(dknrows * sizeof(double));
    udk                         = (double *) malloc(n * sizeof(double));
    boundary                    = (int *) malloc(dknrows * sizeof(int));
    reduced_cols                = (int *) malloc(dknrows * sizeof(int));
    dk                          = (cs *) malloc(1 * sizeof(cs));

    dk->nzmax = dknnz;
    dk->m = dknrows;
    dk->n = dkncols;
    dk->p = dkcols;
    dk->i = dkrows;
    dk->x = dkvals;
    dk->nz = dknnz;
    temp = dk;
    dk = cs_triplet(dk);
    free(temp);

    /* Cache the hessian and the Dk.dot(y) result */
    wdkT = cs_transpose(dk, 1);
    for (i = 0; i < dknnz; i++){ wdkT->x[i] /= w[wdkT->i[i]]; }
    hessian = cs_multiply(dk, wdkT);
    reduced_hessian = cs_spalloc(hessian->m, hessian->n, hessian->nzmax, 1, 0);
    cs_dot_vec(dk, y, dky);

    step = 0;
    cur_converge = rel_tol + 1;
    step_size = 1.0;
    
    /* Perform the Projected Newton iterations until convergence */
    while(step < maxsteps && cur_converge > rel_tol)
    {
        /* Calculate the complete gradient */
        cs_dot_vec(hessian, u, gradient);
        vec_minus_vec(dknrows, gradient, dky, gradient);
        
        /* Calculate the boundary set and get the reduced gradient */
        for (nfree = 0, i = 0; i < dknrows; i++){
            boundary[i] = (u[i] == lam && gradient[i] < -rel_tol) || (u[i] == -lam && gradient[i] > -rel_tol);
            if (!boundary[i]){
                reduced_cols[i] = nfree;
                reduced_gradient[nfree] = gradient[i];
                nfree++;
            }
        }
        
        /* Count the number of elements in the reduced hessian */
        reduced_hessian->m = reduced_hessian->n = nfree;
        for (nfree = 0, i = 0; i < hessian->m; i++){
            if (boundary[i]){ continue; } /* skip the whole column if it's on the boundary */
            reduced_hessian->p[reduced_cols[i]] = nfree; /* set the starting point for this column */
            /* Populate the rows and values */
            for (j = hessian->p[i]; j < hessian->p[i+1]; j++){
                if (!boundary[hessian->i[j]]) {
                    reduced_hessian->i[nfree] = reduced_cols[hessian->i[j]];
                    reduced_hessian->x[nfree] = hessian->x[j];
                    nfree++;
                }
            }
        }
        reduced_hessian->p[reduced_hessian->n] = nfree;

        /* Compute the reduced direction via conjugate gradient */
        j = conjugate_gradient(reduced_hessian, reduced_gradient, reduced_direction, rel_tol);

        /* Take the projected Newton step */
        for (i = 0; i < dknrows; i++){
            if (!boundary[i]){
                u[i] = MIN(lam, MAX(-lam, u[i] - step_size * reduced_direction[reduced_cols[i]]));
            }
        }

        /* Calculate the convergence criteria */
        vec_dot_cs(reduced_gradient, reduced_hessian, reduced_direction);
        cur_converge = fabs(vec_sum(reduced_hessian->m, reduced_direction)) / 2.0;

        /* Hack to handle odd k converging prematurely */
        if (step < 1)
            cur_converge = 1 + rel_tol;

        step_size *= 0.99;
        step++;
    }

    /* beta = y - (1.0 / w) * u.dot(Dk) */
    vec_dot_cs(u, dk, udk);
    for (i = 0; i < n; i++){ udk[i] /= w[i]; }
    vec_minus_vec(n, y, udk, beta);

    free(boundary);
    free(reduced_cols);
    free(gradient);
    free(reduced_gradient);
    free(reduced_direction);
    free(dky);
    free(udk);
    free(dk);
    cs_spfree(wdkT);
    cs_spfree(hessian);
    cs_spfree(reduced_hessian);
    return step;
}


int conjugate_gradient(cs *A, double *b, double *x, double rel_tol)
{
    int i;
    int n;
    int step;
    double alpha;
    double beta;
    double rdotr;
    double rdotr_prev;
    double *r;
    double *rprev;
    double *p;
    double *Ap;

    n = A->m;

    r            = (double *) malloc(n * sizeof(double));
    p            = (double *) malloc(n * sizeof(double));
    Ap           = (double *) malloc(n * sizeof(double));

    /* Initialize the data structures */
    for (i = 0; i < n; i++){ r[i] = p[i] = b[i]; x[i] = 0; }
    rdotr = vec_dot_vec(n, r, r);

    for (step = 0; step < n; step++){
        rdotr_prev = rdotr;
        cs_dot_vec(A, p, Ap);
        alpha = rdotr / vec_dot_vec(n, p, Ap);
        for (i = 0; i < n; i++){
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        /* Check if we're close enough */
        if (vec_norm(n, r) <= rel_tol){ break; }

        rdotr = vec_dot_vec(n, r, r);
        beta = rdotr / rdotr_prev;

        for (i = 0; i < n; i++){ p[i] = r[i] + beta * p[i]; }
    }

    free(r);
    free(p);
    free(Ap);

    return step;
}































