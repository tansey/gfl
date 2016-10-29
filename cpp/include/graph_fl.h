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

#ifndef GRAPH_FL_H
#define GRAPH_FL_H

#include <math.h>
#include "tf.h"
#include "utils.h"

#define VARYING_PENALTY_DELAY 50


int graph_fused_lasso (int n, double *y,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta);

int graph_fused_lasso_weight (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta);

int graph_fused_lasso_logit (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta);

int graph_fused_lasso_warm (int n, double *y,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u);

int graph_fused_lasso_weight_warm (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u);

int graph_fused_lasso_logit_warm (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u);

/**********************************************************************
Lambda-per-edge case where all trails are required to be of length 1.
*********************************************************************/
int graph_fused_lasso_lams (int n, double *y,
                            int ntrails, int *trails, int *breakpoints,
                            double *lam, double alpha, double inflate,
                            int maxsteps, double converge,
                            double *beta);

int graph_fused_lasso_lams_weight (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta);

int graph_fused_lasso_lams_logit (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta);

int graph_fused_lasso_lams_warm (int n, double *y,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u);

int graph_fused_lasso_lams_weight_warm (int n, double *y, double *w,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u);

int graph_fused_lasso_logit_lams_warm (int n, int *trials, int *successes,
                        int ntrails, int *trails, int *breakpoints,
                        double *lam, double alpha, double inflate,
                        int maxsteps, double converge,
                        double *beta, double *z, double *u);


void update_beta(int n, double *y, double *z, double *u, int *nzmap, int *zmap, double alpha, double *beta);
void update_beta_weight(int n, double *y, double *w, double *z, double *u, int *nzmap, int *zmap, double alpha, double *beta);
void update_z(int ntrails, int *trails, int *breakpoints, double *beta, double *u, double lam, double *ybuf, double *wbuf, double *tf_dp_buf, double *z);
void update_u(int n, double *beta, double *z, int *zmap, int *nzmap, double *u);
double primal_resnorm(int n, double *beta, double *z, int *nzmap, int *zmap);
double dual_resnorm(int nz, double *z, double *zold, double alpha);
void update_z_lams(int ntrails, int *trails, int *breakpoints, double *beta, double *u, double *lam, double *ybuf, double *wbuf, double *tf_dp_buf, double *z);

#endif













