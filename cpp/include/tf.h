/****************************************************************************
 * Copyright (C) 2014 by Taylor Arnold, Veeranjaneyulu Sadhanala,           *
 *                       Ryan Tibshirani                                    *
 *                                                                          *
 * This modified file is released under the MIT license as long as it is    *
 * included and used within the gfl library / package. The authors of gfl   *
 * make no copyright claims to this portion of the code and provide no      *
 * guarantees. If you are using this code for anything other than the gfl   *
 * library / package, you are subject to the license for the original       *
 * glmgen library / package (see below).                                    *
 *                                                                          *
 * This file is part of the glmgen library / package.                       *
 *                                                                          *
 *   glmgen is free software: you can redistribute it and/or modify it      *
 *   under the terms of the GNU Lesser General Public License as published  *
 *   by the Free Software Foundation, either version 2 of the License, or   *
 *   (at your option) any later version.                                    *
 *                                                                          *
 *   glmgen is distributed in the hope that it will be useful,              *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 *   GNU Lesser General Public License for more details.                    *
 *                                                                          *
 *   You should have received a copy of the GNU Lesser General Public       *
 *   License along with glmgen. If not, see <http://www.gnu.org/licenses/>. *
 ****************************************************************************/

/**
 * @file tf.h
 * @author Taylor Arnold, Ryan Tibshirani, Veerun Sadhanala
 * @date 2014-12-23
 * @brief Main functions for fitting trend filtering models.
 *
 * Here.
 */

#ifndef TF_H
#define TF_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "utils.h"

/* Dynamic programming routines */

#ifdef __CUDACC__
 __device__ void tf_dp (int n, double *y, double lam, double *beta, 
                        double *x, double *a, double *b, double *tm, double *tp);
#else
 void tf_dp (int n, double *y, double lam, double *beta,
            double *x, double *a, double *b, double *tm, double *tp);
#endif

#ifdef __CUDACC__
 __device__ void tf_dp_weight (int n, double *y, double *w, double lam, double *beta, 
                                double *x, double *a, double *b, double *tm, double *tp);
#else
 void tf_dp_weight (int n, double *y, double *w, double lam, double *beta,
                    double *x, double *a, double *b, double *tm, double *tp);
#endif
#endif
