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

#ifndef GRAPH_TF_H
#define GRAPH_TF_H

#include "csparse.h"
#include "utils.h"

int graph_trend_filtering_weight_warm (int n, double *y, double *w, double lam,
                                       int dknrows, int dkncols, int dknnz, int *dkrows, int *dkcols, double *dkvals,
                                       int maxsteps, double converge,
                                       double *beta, double *u);

int conjugate_gradient(cs *A, double *b, double *x, double rel_tol);
#endif


