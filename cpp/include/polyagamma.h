/****************************************************************************
 * Copyright (C) 2016 by Wesley Tansey                                      *
 *                                                                          *
 * This file is part of the the GFL library / package. This is a C          *
 * translation of the HelloPG/PolyaGamma.h file available at                *
 * https://github.com/jgscott/helloPG                                       *
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

#ifndef POLYAGAMMA_H
#define POLYAGAMMA_H

#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>

/* The numerical accuracy of __PI will affect your distribution. */
#define __PI 3.141592653589793238462643383279502884197
#define HALFPISQ (0.5 * __PI * __PI)
#define FOURPISQ (4 * __PI * __PI)
#define __TRUNC (0.64)
#define __TRUNC_RECIP (1.0 / __TRUNC)

/* Draw. */
double polyagamma_draw(int n, double z, const gsl_rng *random);
double polyagamma_draw_like_devroye(double z, const gsl_rng *random);

/* Helper. */
double polyagamma_a(int n, double x);
double polyagamma_pigauss(double x, double Z);
double polyagamma_mass_texpon(double Z);
double polyagamma_rtigauss(double Z, const gsl_rng *random);
double polyagamma_p_norm (double x, int use_log);
double polyagamma_expon_rate (double rate, const gsl_rng *random);

#endif

