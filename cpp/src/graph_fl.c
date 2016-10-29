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
 
 #include "graph_fl.h"

int graph_fused_lasso (int n, double *y,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta)
{
    return graph_fused_lasso_weight(n, y, (double*)0, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta);
}


int graph_fused_lasso_weight (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta)
{
    int i;
    int nz;
    double *z;
    double *u;
    
    nz = breakpoints[ntrails-1];
    z = (double *) malloc(nz * sizeof(double));
    u = (double *) malloc(nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < nz; i++)
    {
        z[i] = 0;
        u[i] = 0;
    }

    i = graph_fused_lasso_weight_warm(n, y, w, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta, z, u);

    free(z);
    free(u);

    return i;
}

int graph_fused_lasso_warm (int n, double *y,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u)
{
    return graph_fused_lasso_weight_warm(n, y, (double*)0, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta, z, u);
}

int graph_fused_lasso_weight_warm (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u)
{
    int i;
    int step;
    int nz;
    int ti;
    int zmi;
    int wbufsize;
    double cur_converge;
    double presnorm;
    double dresnorm;
    double *ztemp;
    double *zold;
    int *zmap;
    int *nzmap;
    int *zmapbreaks;
    int *zmapoffsets;
    double *z_ybuff;
    double *z_wbuff;
    double *z_ptr;
    double *tf_dp_buf;

    nz = breakpoints[ntrails-1];
    z_ptr = z;

    zold            = (double *) malloc(nz * sizeof(double)); /* Use a double buffering strategy for z */
    zmap            = (int *) malloc(nz * sizeof(int));
    nzmap           = (int *) malloc(n * sizeof(int));
    zmapbreaks      = (int *) malloc(n * sizeof(int));
    zmapoffsets     = (int *) malloc(n * sizeof(int));
    tf_dp_buf       = (double *) malloc((nz * 8 - 2 * ntrails) * sizeof(double));

    memcpy(zold, z, nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < n; i++)
    {
        zmap[i] = 0;
        nzmap[i] = 0;
    }
    
    /* Find the largest amount of memory we need to allocate for the z-update buffers */
    wbufsize = breakpoints[0];
    for (i = 1; i < ntrails; i++)
    {
        if (breakpoints[i]-breakpoints[i-1] > wbufsize) { wbufsize = breakpoints[i]-breakpoints[i-1]; }
    }
    z_wbuff         = (double *) malloc(wbufsize * sizeof(double));
    z_ybuff         = (double *) malloc(wbufsize * sizeof(double));

    /* weight for each fused lasso */
    for (i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; }

    /* Create map from z to beta */
    for (i = 0; i < nz; i++){ nzmap[trails[i]]++; } /* number of repetitions of beta_i */
    zmapbreaks[0] = zmapoffsets[0] = nzmap[0];
    for (i = 1; i < n; i++)
    {
        zmapbreaks[i] = zmapbreaks[i-1] + nzmap[i];
        zmapoffsets[i] = nzmap[i];
    }
    for (i = 0; i < nz; i++) /* create the map */
    {
        ti = trails[i];
        zmi = zmapbreaks[ti] - zmapoffsets[ti];
        zmap[zmi] = i;
        zmapoffsets[ti]--;
    }
    
    step = 0;
    cur_converge = converge + 1;

    /* Perform the ADMM iterations until convergence */
    while(step < maxsteps && cur_converge > converge)
    {
        /* Update beta */
        if (w)
            update_beta_weight(n, y, w, z, u, nzmap, zmap, alpha, beta);
        else
            update_beta(n, y, z, u, nzmap, zmap, alpha, beta);

        /* swap the z buffers */
        ztemp = z;
        z = zold;
        zold = ztemp;
        
        /* Update each trail dual variable */
        update_z(ntrails, trails, breakpoints, beta, u, lam, z_ybuff, z_wbuff, tf_dp_buf, z);
        
        /* Update the scaled dual variable */
        update_u(n, beta, z, zmap, nzmap, u);

        /* Update the convergence diagnostics */
        presnorm = primal_resnorm(n, beta, z, nzmap, zmap);
        dresnorm = dual_resnorm(nz, z, zold, alpha);
        cur_converge = MAX(presnorm, dresnorm);

        /* Varying penalty parameter */
        if (step % VARYING_PENALTY_DELAY == 0 && presnorm > 10 * dresnorm)
        {
            alpha *= inflate;
            for(i = 0; i < nz; i++){ u[i] /= inflate; }
            for(i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; } /* weight for each fused lasso */
        }
        else if (step % VARYING_PENALTY_DELAY == 0 && dresnorm > 10 * presnorm)
        {
            alpha /= inflate;
            for(i = 0; i < nz; i++){ u[i] *= inflate; }
            for(i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; } /* weight for each fused lasso */
        }

        step++;
    }

    /* Make sure to return the final z to the user */
    if (z_ptr == zold)
    {
        ztemp = z;
        z = zold;
        zold = ztemp;
    }

    /* free up the resources */
    free(zold);
    free(zmap);
    free(nzmap);
    free(zmapbreaks);
    free(zmapoffsets);
    free(z_ybuff);
    free(z_wbuff);
    free(tf_dp_buf);
    
    return step;
}



int graph_fused_lasso_logit (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta)
{
    int i;
    int nz;
    double p;
    double *z;
    double *u;
    
    nz = breakpoints[ntrails-1];
    z = (double *) malloc(nz * sizeof(double));
    u = (double *) malloc(nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < nz; i++)
    {
        z[i] = 0;
        u[i] = 0;
    }

    /* Pre-smoothing for an initial guess */
    for (i = 0; i < n; i++)
    {
        p = (successes[i] + 1.0) / (trials[i] + 2.0);
        beta[i] = log(p / (1.0-p));
    }

    i = graph_fused_lasso_logit_warm(n, trials, successes, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta, z, u);

    free(z);
    free(u);

    return i;
}

int graph_fused_lasso_logit_warm (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u)
{
    int i;
    int step;
    int nz;
    int ti;
    int zmi;
    int wbufsize;
    double cur_converge;
    double presnorm;
    double dresnorm;
    double p;
    double *zold;
    double *ztemp;
    double *w;
    double *y;
    int *zmap;
    int *nzmap;
    int *zmapbreaks;
    int *zmapoffsets;
    double *z_ybuff;
    double *z_wbuff;
    double *z_ptr;
    double *tf_dp_buf;

    nz = breakpoints[ntrails-1];
    z_ptr = z;

    /*z               = (double *) malloc(nz * sizeof(double));*/
    zold            = (double *) malloc(nz * sizeof(double)); /* Use a double buffering strategy for z */
    zmap            = (int *) malloc(nz * sizeof(int));
    nzmap           = (int *) malloc(n * sizeof(int));
    zmapbreaks      = (int *) malloc(n * sizeof(int));
    zmapoffsets     = (int *) malloc(n * sizeof(int));
    w               = (double *) malloc(n * sizeof(double));
    y               = (double *) malloc(n * sizeof(double));
    tf_dp_buf       = (double *) malloc((nz * 8 - 2 * ntrails) * sizeof(double));
    
    memcpy(zold, z, nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < n; i++)
    {
        zmap[i] = 0;
        nzmap[i] = 0;
    }
    
    /* Find the largest amount of memory we need to allocate for the z-update buffers */
    wbufsize = breakpoints[0];
    for (i = 1; i < ntrails; i++)
    {
        if (breakpoints[i]-breakpoints[i-1] > wbufsize) { wbufsize = breakpoints[i]-breakpoints[i-1]; }
    }
    z_ybuff         = (double *) malloc(wbufsize * sizeof(double));
    z_wbuff         = (double *) malloc(wbufsize * sizeof(double));
    
    /* weight for each fused lasso */
    for (i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; }
    
    /* Create map from z to beta */
    for (i = 0; i < nz; i++){ nzmap[trails[i]]++; } /* number of repetitions of beta_i */
    zmapbreaks[0] = zmapoffsets[0] = nzmap[0];
    for (i = 1; i < n; i++)
    {
        zmapbreaks[i] = zmapbreaks[i-1] + nzmap[i];
        zmapoffsets[i] = nzmap[i];
    }
    
    for (i = 0; i < nz; i++) /* create the map */
    {
        ti = trails[i];
        zmi = zmapbreaks[ti] - zmapoffsets[ti];
        zmap[zmi] = i;
        zmapoffsets[ti]--;
    }
    
    step = 0;
    cur_converge = converge + 1;
    
    /* Perform the ADMM iterations until convergence */
    while(step < maxsteps && cur_converge > converge)
    {
        /* Perform a second-order Taylor expansion */
        for (i = 0; i < n; i++)
        {
            p = 1.0 / (1.0 + exp(-beta[i]));
            w[i] = trials[i] * p * (1.0 - p) + 1e-12;
            y[i] = beta[i] - (trials[i] * p - successes[i]) / w[i];
        }
        
        /* Update beta */
        update_beta_weight(n, y, w, z, u, nzmap, zmap, alpha, beta);
        
        /* swap the z buffers */
        ztemp = z;
        z = zold;
        zold = ztemp;
        
        /* Update each trail dual variable */
        update_z(ntrails, trails, breakpoints, beta, u, lam, z_ybuff, z_wbuff, tf_dp_buf, z);
        
        /* Update the scaled dual variable */
        update_u(n, beta, z, zmap, nzmap, u);
        
        /* Update the convergence diagnostics */
        presnorm = primal_resnorm(n, beta, z, nzmap, zmap);
        dresnorm = dual_resnorm(nz, z, zold, alpha);
        cur_converge = MAX(presnorm, dresnorm);

        /* Varying penalty parameter */
        if (step % VARYING_PENALTY_DELAY == 0 && presnorm > 10 * dresnorm)
        {
            alpha *= inflate;
            for(i = 0; i < nz; i++){ u[i] /= inflate; }
        }
        else if (step % VARYING_PENALTY_DELAY == 0 && dresnorm > 10 * presnorm)
        {
            alpha /= inflate;
            for(i = 0; i < nz; i++){ u[i] *= inflate; }
        }

        step++;
    }
    
    /* Make sure to return the final z to the user */
    if (z_ptr == zold)
    {
        ztemp = z;
        z = zold;
        zold = ztemp;
    }

    /* free up the resources */
    free(zold);
    free(zmap);
    free(nzmap);
    free(zmapbreaks);
    free(zmapoffsets);
    free(z_ybuff);
    free(z_wbuff);
    free(w);
    free(y);
    free(tf_dp_buf);

    return step;
}

void update_beta(int n, double *y, double *z, double *u, int *nzmap, int *zmap, double alpha, double *beta)
{
    int i;
    int ridx;
    int zmap_idx;
    double r;

    zmap_idx = 0;

    /* Update each beta in closed form */
    for (i = 0; i < n; i++)
    {
        r = 0;    
        /* Sum the dual terms */
        for (ridx = 0; ridx < nzmap[i]; ridx++, zmap_idx++)
        {
            r += z[zmap[zmap_idx]] - u[zmap[zmap_idx]];
        }
        beta[i] = (2.0 * y[i] + alpha * r) / (2.0 + alpha * nzmap[i]);
    }
}

void update_beta_weight(int n, double *y, double *w, double *z, double *u, int *nzmap, int *zmap, double alpha, double *beta)
{
    int i;
    int ridx;
    int zmap_idx;
    double r;

    zmap_idx = 0;

    /* Update each beta in closed form */
    for (i = 0; i < n; i++)
    {
        r = 0;    
        /* Sum the dual terms */
        for (ridx = 0; ridx < nzmap[i]; ridx++, zmap_idx++)
        {
            r += z[zmap[zmap_idx]] - u[zmap[zmap_idx]];
        }
        beta[i] = (2.0 * y[i] * w[i] + alpha * r) / (2.0 * w[i] + alpha * nzmap[i]);
    }
}


void update_z(int ntrails, int *trails, int *breakpoints, double *beta, double *u, double lam, double *ybuf, double *wbuf, double *tf_dp_buf, double *z)
{
    int i;
    int j;
    int trailstart;
    int trailend;
    int trailsize;
    double *x;
    double *a;
    double *b;
    double *tm;
    double *tp;
    
    trailstart = 0;
    
    /* Update each trail via a 1-d fused lasso. */
    for (i = 0; i < ntrails; i++)
    {
        trailend = breakpoints[i];
        trailsize = trailend - trailstart;

        /* Calculate the trail y values: (beta + u) */
        for (j = trailstart; j < trailend; j++) { ybuf[j-trailstart] = beta[trails[j]] + u[j]; }

        /* Calculate the starts of this z's buffers */
        x = tf_dp_buf + 8*trailstart - 2*i;
        a = x + 2*trailsize;
        b = x + 4*trailsize;
        tm = x + 6*trailsize;
        tp = x + 7*trailsize - 1;
        
        tf_dp_weight(trailsize, ybuf, wbuf, lam, z + trailstart,
                        x, a, b, tm, tp);
        
        trailstart = trailend;
    }
}

void update_u(int n, double *beta, double *z, int *zmap, int *nzmap, double *u)
{
    int i;
    int zmoffset;
    int zmi;
    double b;

    zmoffset = 0;
    /* Update the dual variable with the primal residual */
    for (i = 0; i < n; i++)
    {
        b = beta[i];

        for (zmi = zmoffset; zmi < nzmap[i]+zmoffset; zmi++)
            u[zmap[zmi]] += b - z[zmap[zmi]];

        zmoffset += nzmap[i];
    }
}

double primal_resnorm(int n, double *beta, double *z, int *nzmap, int *zmap)
{
    int i;
    int ridx;
    int zmap_idx;
    double r;
    double ri; 

    r = 0;
    zmap_idx = 0;
    for(i=0; i < n; i++)
    {
        ri = beta[i];

        /* Subtract all the dual terms */
        for (ridx = 0; ridx < nzmap[i]; ridx++, zmap_idx++)
            r += (ri - z[zmap[zmap_idx]]) * (ri - z[zmap[zmap_idx]]);
    }
    return sqrt(r / n);
}

double dual_resnorm(int nz, double *z, double *zold, double alpha)
{
    int i;
    double r;

    r = 0;

    for (i=0; i < nz; i++)
    {
        r += (alpha * (z[i] - zold[i])) * (alpha * (z[i] - zold[i]));
    }
    return sqrt(r / nz);
}

/**********************************************************************
All code below deals with the lambda-per-edge case where all trails are
required to be of length 1.
***********************************************************************/

int graph_fused_lasso_lams_weight (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta)
{
    int i;
    int nz;
    double *z;
    double *u;
    
    nz = breakpoints[ntrails-1];
    z = (double *) malloc(nz * sizeof(double));
    u = (double *) malloc(nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < nz; i++)
    {
        z[i] = 0;
        u[i] = 0;
    }

    i = graph_fused_lasso_lams_weight_warm(n, y, w, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta, z, u);

    free(z);
    free(u);

    return i;
}

int graph_fused_lasso_lams_warm (int n, double *y,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u)
{
    return graph_fused_lasso_lams_weight_warm(n, y, (double*)0, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta, z, u);
}

int graph_fused_lasso_lams (int n, double *y,
                        int ntrails, int *trails, int *breakpoints,
                        double* lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta)
{
    return graph_fused_lasso_lams_weight(n, y, (double*)0, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta);
}

int graph_fused_lasso_lams_weight_warm (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u)
{
    int i;
    int step;
    int nz;
    int ti;
    int zmi;
    int wbufsize;
    double cur_converge;
    double presnorm;
    double dresnorm;
    double *ztemp;
    double *zold;
    int *zmap;
    int *nzmap;
    int *zmapbreaks;
    int *zmapoffsets;
    double *z_ybuff;
    double *z_wbuff;
    double *z_ptr;
    double *tf_dp_buf;

    nz = breakpoints[ntrails-1];
    z_ptr = z;

    if (breakpoints[ntrails-1] != (ntrails*2))
    {
        printf("All trails must be of length 1 for the multiple-lambda case.\n");
        exit(EXIT_FAILURE);
    }

    zold            = (double *) malloc(nz * sizeof(double)); /* Use a double buffering strategy for z */
    zmap            = (int *) malloc(nz * sizeof(int));
    nzmap           = (int *) malloc(n * sizeof(int));
    zmapbreaks      = (int *) malloc(n * sizeof(int));
    zmapoffsets     = (int *) malloc(n * sizeof(int));
    tf_dp_buf       = (double *) malloc((nz * 8 - 2 * ntrails) * sizeof(double));

    memcpy(zold, z, nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < n; i++)
    {
        zmap[i] = 0;
        nzmap[i] = 0;
    }
    
    /* Find the largest amount of memory we need to allocate for the z-update buffers */
    wbufsize = breakpoints[0];
    for (i = 1; i < ntrails; i++)
    {
        if (breakpoints[i]-breakpoints[i-1] > wbufsize) { wbufsize = breakpoints[i]-breakpoints[i-1]; }
    }
    z_wbuff         = (double *) malloc(wbufsize * sizeof(double));
    z_ybuff         = (double *) malloc(wbufsize * sizeof(double));

    /* weight for each fused lasso */
    for (i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; }

    /* Create map from z to beta */
    for (i = 0; i < nz; i++){ nzmap[trails[i]]++; } /* number of repetitions of beta_i */
    zmapbreaks[0] = zmapoffsets[0] = nzmap[0];
    for (i = 1; i < n; i++)
    {
        zmapbreaks[i] = zmapbreaks[i-1] + nzmap[i];
        zmapoffsets[i] = nzmap[i];
    }
    for (i = 0; i < nz; i++) /* create the map */
    {
        ti = trails[i];
        zmi = zmapbreaks[ti] - zmapoffsets[ti];
        zmap[zmi] = i;
        zmapoffsets[ti]--;
    }
    
    step = 0;
    cur_converge = converge + 1;

    /* Perform the ADMM iterations until convergence */
    while(step < maxsteps && cur_converge > converge)
    {
        /* Update beta */
        if (w)
            update_beta_weight(n, y, w, z, u, nzmap, zmap, alpha, beta);
        else
            update_beta(n, y, z, u, nzmap, zmap, alpha, beta);

        /* swap the z buffers */
        ztemp = z;
        z = zold;
        zold = ztemp;
        
        /* Update each trail dual variable */
        update_z_lams(ntrails, trails, breakpoints, beta, u, lam, z_ybuff, z_wbuff, tf_dp_buf, z);
        
        /* Update the scaled dual variable */
        update_u(n, beta, z, zmap, nzmap, u);

        /* Update the convergence diagnostics */
        presnorm = primal_resnorm(n, beta, z, nzmap, zmap);
        dresnorm = dual_resnorm(nz, z, zold, alpha);
        cur_converge = MAX(presnorm, dresnorm);

        /* Varying penalty parameter */
        if (step % VARYING_PENALTY_DELAY == 0 && presnorm > 10 * dresnorm)
        {
            alpha *= inflate;
            for(i = 0; i < nz; i++){ u[i] /= inflate; }
            for(i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; } /* weight for each fused lasso */
        }
        else if (step % VARYING_PENALTY_DELAY == 0 && dresnorm > 10 * presnorm)
        {
            alpha /= inflate;
            for(i = 0; i < nz; i++){ u[i] *= inflate; }
            for(i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; } /* weight for each fused lasso */
        }

        step++;
    }

    /* Make sure to return the final z to the user */
    if (z_ptr == zold)
    {
        ztemp = z;
        z = zold;
        zold = ztemp;
    }

    /* free up the resources */
    free(zold);
    free(zmap);
    free(nzmap);
    free(zmapbreaks);
    free(zmapoffsets);
    free(z_ybuff);
    free(z_wbuff);
    free(tf_dp_buf);
    
    return step;
}

int graph_fused_lasso_logit_lams (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta)
{
    int i;
    int nz;
    double p;
    double *z;
    double *u;
    
    nz = breakpoints[ntrails-1];
    z = (double *) malloc(nz * sizeof(double));
    u = (double *) malloc(nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < nz; i++)
    {
        z[i] = 0;
        u[i] = 0;
    }

    /* Pre-smoothing for an initial guess */
    for (i = 0; i < n; i++)
    {
        p = (successes[i] + 1.0) / (trials[i] + 2.0);
        beta[i] = log(p / (1.0-p));
    }

    i = graph_fused_lasso_logit_lams_warm(n, trials, successes, ntrails, trails, breakpoints, lam, alpha, inflate, maxsteps, converge, beta, z, u);

    free(z);
    free(u);

    return i;
}

int graph_fused_lasso_logit_lams_warm (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u)
{
    int i;
    int step;
    int nz;
    int ti;
    int zmi;
    int wbufsize;
    double cur_converge;
    double presnorm;
    double dresnorm;
    double p;
    double *zold;
    double *ztemp;
    double *w;
    double *y;
    int *zmap;
    int *nzmap;
    int *zmapbreaks;
    int *zmapoffsets;
    double *z_ybuff;
    double *z_wbuff;
    double *z_ptr;
    double *tf_dp_buf;

    nz = breakpoints[ntrails-1];
    z_ptr = z;

    if (breakpoints[ntrails-1] != (ntrails*2))
    {
        printf("All trails must be of length 1 for the multiple-lambda case.\n");
        exit(EXIT_FAILURE);
    }

    /*z               = (double *) malloc(nz * sizeof(double));*/
    zold            = (double *) malloc(nz * sizeof(double)); /* Use a double buffering strategy for z */
    zmap            = (int *) malloc(nz * sizeof(int));
    nzmap           = (int *) malloc(n * sizeof(int));
    zmapbreaks      = (int *) malloc(n * sizeof(int));
    zmapoffsets     = (int *) malloc(n * sizeof(int));
    w               = (double *) malloc(n * sizeof(double));
    y               = (double *) malloc(n * sizeof(double));
    tf_dp_buf       = (double *) malloc((nz * 8 - 2 * ntrails) * sizeof(double));
    
    memcpy(zold, z, nz * sizeof(double));

    /* Zero-out vectors to start */
    for (i = 0; i < n; i++)
    {
        zmap[i] = 0;
        nzmap[i] = 0;
    }
    
    /* Find the largest amount of memory we need to allocate for the z-update buffers */
    wbufsize = breakpoints[0];
    for (i = 1; i < ntrails; i++)
    {
        if (breakpoints[i]-breakpoints[i-1] > wbufsize) { wbufsize = breakpoints[i]-breakpoints[i-1]; }
    }
    z_ybuff         = (double *) malloc(wbufsize * sizeof(double));
    z_wbuff         = (double *) malloc(wbufsize * sizeof(double));
    
    /* weight for each fused lasso */
    for (i = 0; i < wbufsize; i++) { z_wbuff[i] = alpha / 2.0; }
    
    /* Create map from z to beta */
    for (i = 0; i < nz; i++){ nzmap[trails[i]]++; } /* number of repetitions of beta_i */
    zmapbreaks[0] = zmapoffsets[0] = nzmap[0];
    for (i = 1; i < n; i++)
    {
        zmapbreaks[i] = zmapbreaks[i-1] + nzmap[i];
        zmapoffsets[i] = nzmap[i];
    }
    
    for (i = 0; i < nz; i++) /* create the map */
    {
        ti = trails[i];
        zmi = zmapbreaks[ti] - zmapoffsets[ti];
        zmap[zmi] = i;
        zmapoffsets[ti]--;
    }
    
    step = 0;
    cur_converge = converge + 1;
    
    /* Perform the ADMM iterations until convergence */
    while(step < maxsteps && cur_converge > converge)
    {
        /* Perform a second-order Taylor expansion */
        for (i = 0; i < n; i++)
        {
            p = 1.0 / (1.0 + exp(-beta[i]));
            w[i] = trials[i] * p * (1.0 - p) + 1e-12;
            y[i] = beta[i] - (trials[i] * p - successes[i]) / w[i];
        }
        
        /* Update beta */
        update_beta_weight(n, y, w, z, u, nzmap, zmap, alpha, beta);
        
        /* swap the z buffers */
        ztemp = z;
        z = zold;
        zold = ztemp;
        
        /* Update each trail dual variable */
        update_z_lams(ntrails, trails, breakpoints, beta, u, lam, z_ybuff, z_wbuff, tf_dp_buf, z);
        
        /* Update the scaled dual variable */
        update_u(n, beta, z, zmap, nzmap, u);
        
        /* Update the convergence diagnostics */
        presnorm = primal_resnorm(n, beta, z, nzmap, zmap);
        dresnorm = dual_resnorm(nz, z, zold, alpha);
        cur_converge = MAX(presnorm, dresnorm);

        /* Varying penalty parameter */
        if (step % VARYING_PENALTY_DELAY == 0 && presnorm > 10 * dresnorm)
        {
            alpha *= inflate;
            for(i = 0; i < nz; i++){ u[i] /= inflate; }
        }
        else if (step % VARYING_PENALTY_DELAY == 0 && dresnorm > 10 * presnorm)
        {
            alpha /= inflate;
            for(i = 0; i < nz; i++){ u[i] *= inflate; }
        }

        step++;
    }
    
    /* Make sure to return the final z to the user */
    if (z_ptr == zold)
    {
        ztemp = z;
        z = zold;
        zold = ztemp;
    }

    /* free up the resources */
    free(zold);
    free(zmap);
    free(nzmap);
    free(zmapbreaks);
    free(zmapoffsets);
    free(z_ybuff);
    free(z_wbuff);
    free(w);
    free(y);
    free(tf_dp_buf);

    return step;
}

void update_z_lams(int ntrails, int *trails, int *breakpoints, double *beta, double *u, double *lam, double *ybuf, double *wbuf, double *tf_dp_buf, double *z)
{
    int i;
    int j;
    int trailstart;
    int trailend;
    int trailsize;
    double *x;
    double *a;
    double *b;
    double *tm;
    double *tp;
    
    trailstart = 0;
    
    /* Update each trail via a 1-d fused lasso. */
    for (i = 0; i < ntrails; i++)
    {
        trailend = breakpoints[i];
        trailsize = trailend - trailstart;

        /* Calculate the trail y values: (beta + u) */
        for (j = trailstart; j < trailend; j++) { ybuf[j-trailstart] = beta[trails[j]] + u[j]; }

        /* Calculate the starts of this z's buffers */
        x = tf_dp_buf + 8*trailstart - 2*i;
        a = x + 2*trailsize;
        b = x + 4*trailsize;
        tm = x + 6*trailsize;
        tp = x + 7*trailsize - 1;
        
        tf_dp_weight(trailsize, ybuf, wbuf, lam[i], z + trailstart,
                        x, a, b, tm, tp);
        
        trailstart = trailend;
    }
}











