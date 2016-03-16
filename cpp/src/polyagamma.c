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

#include "polyagamma.h"

/*//////////////////////////////////////////////////////////////////////////////
                 // Utility //
//////////////////////////////////////////////////////////////////////////////*/

double polyagamma_a(int n, double x)
{
  double K;
  double y;
  double expnt;

  K = (n + 0.5) * __PI;
  if (x > __TRUNC) {
    y = K * gsl_sf_exp( -0.5 * K*K * x );
  }
  else {
    expnt = -1.5 * (gsl_sf_log(0.5 * __PI)  + gsl_sf_log(x)) + gsl_sf_log(K) - 2.0 * (n+0.5)*(n+0.5) / x;
    y = gsl_sf_exp(expnt);
    /* y = pow(0.5 * __PI * x, -1.5) * K * gsl_sf_exp( -2.0 * (n+0.5)*(n+0.5) / x);
    // ^- unstable for small x?*/
  }
  return y;
}

double polyagamma_pigauss(double x, double Z)
{
  double b;
  double a;
  double y;
  b = sqrt(1.0 / x) * (x * Z - 1);
  a = sqrt(1.0 / x) * (x * Z + 1) * -1.0;
  y = polyagamma_p_norm(b,0) + gsl_sf_exp(2 * Z) * polyagamma_p_norm(a,0);
  return y;
}

double polyagamma_mass_texpon(double Z)
{
  double t;
  double fz;
  double b;
  double a;
  double x0;
  double xb;
  double xa;
  double qdivp;

  t = __TRUNC;

  fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
  b = sqrt(1.0 / t) * (t * Z - 1);
  a = sqrt(1.0 / t) * (t * Z + 1) * -1.0;

  x0 = gsl_sf_log(fz) + fz * t;
  xb = x0 - Z + polyagamma_p_norm(b, 1);
  xa = x0 + Z + polyagamma_p_norm(a, 1);

  qdivp = 4 / __PI * ( gsl_sf_exp(xb) + gsl_sf_exp(xa) );

  return 1.0 / (1.0 + qdivp);
}

double polyagamma_rtigauss(double Z, const gsl_rng *random)
{
  double t;
  double X;
  double alpha;
  double E1;
  double E2;
  double mu;
  double Y;
  double half_mu;
  double mu_Y;

  Z = fabs(Z);
  t = __TRUNC;
  X = t + 1.0;
  if (__TRUNC_RECIP > Z) { /* mu > t */
    alpha = 0.0;
    while (gsl_rng_uniform(random) > alpha) {
      /* X = t + 1.0;
      // while (X > t)
      //    X = 1.0 / gsl_ran_gamma(0.5, 1.0 / 0.5);
      // Slightly faster to use truncated normal. */
      E1 = polyagamma_expon_rate(1.0, random); E2 = polyagamma_expon_rate(1.0, random);
      while ( E1*E1 > 2 * E2 / t) {
    E1 = polyagamma_expon_rate(1.0, random); E2 = polyagamma_expon_rate(1.0, random);
      }
      X = 1 + E1 * t;
      X = t / (X * X);
      alpha = gsl_sf_exp(-0.5 * Z*Z * X);
    }
  }
  else {
    mu = 1.0 / Z;
    while (X > t) {
      Y = gsl_ran_gaussian(random, 1.0); Y *= Y;
      half_mu = 0.5 * mu;
      mu_Y    = mu  * Y;
      X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      if (gsl_rng_uniform(random) > mu / (mu + X))
    X = mu*mu / X;
    }
  }
  return X;
}

/*//////////////////////////////////////////////////////////////////////////////
                  // Sample //
//////////////////////////////////////////////////////////////////////////////*/

double polyagamma_draw(int n, double z, const gsl_rng *random)
{
    double sum;
    int i;

  if (n < 1) {
    printf("PolyaGamma::draw: n < 1.");
    exit(1);
  }
  sum = 0.0;
  for (i = 0; i < n; ++i)
    sum += polyagamma_draw_like_devroye(z, random);
  return sum;
}

double polyagamma_draw_like_devroye(double Z, const gsl_rng *random)
{
  double fz;
  double X;
  double S;
  double Y;
  int n;
  int go;

  /* Change the parameter. */
  Z = fabs(Z) * 0.5;

  /* Now sample 0.25 * J^*(1, Z := Z/2). */
  fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
  /* ... Problems with large Z?  Try using q_over_p.
  // double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
  // double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z); */

  X = 0.0;
  S = 1.0;
  Y = 0.0;

  while (1) {

    /* if (gsl_rng_uniform(random) < p/(p+q)) */
    if ( gsl_rng_uniform(random) < polyagamma_mass_texpon(Z) )
      X = __TRUNC + polyagamma_expon_rate(1.0, random) / fz;
    else
      X = polyagamma_rtigauss(Z, random);

    S = polyagamma_a(0, X);
    Y = gsl_rng_uniform(random) * S;
    n = 0;
    go = 1;

    while (go) {
      ++n;
      if (n%2==1) {
    S = S - polyagamma_a(n, X);
    if ( Y<=S ) return 0.25 * X;
      }
      else {
    S = S + polyagamma_a(n, X);
    if ( Y>S ) go = 0;
      }
    }

    /* Need Y <= S in event that Y = S, e.g. when X = 0. */
  }
}

double polyagamma_p_norm(double x, int use_log)
{
  double m;
  m = gsl_cdf_ugaussian_P(x);
  if (use_log) m = gsl_sf_log(m);
  return m;
}

double polyagamma_expon_rate (double rate, const gsl_rng *random)
{
    return gsl_ran_exponential(random, 1.0 / rate);
}
