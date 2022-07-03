/*
 * LeastSquares.h - routines for 2D and 3D least squares
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#include "wlspoly.hpp"

#ifdef USE_VC
#include <Vc/Vc>
#endif

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>


#ifdef USE_VC
template <class S> using Vector = std::vector<S, Vc::Allocator<S>>;
#else
template <class S> using Vector = std::vector<S>;
#endif


//
// Approximate a spatial derivative from a number of irregularly-spaced points
//
template <class S, class A>
A least_squares_val(const S xt, const S yt,
                    const Vector<S>& x, const Vector<S>& y,
                    const Vector<A>& u,
                    const size_t istart, const size_t iend) {

    //printf("  target point at %g %g %g\n", xt, yt, zt);

    // prepare the arrays for CxxPolyFit
    std::vector<S> xs(2*(iend-istart));
    std::vector<S> vs(iend-istart);

    // fill the arrays
    size_t icnt = 0;
    for (size_t i=0; i<iend-istart; ++i) {
        // eventually want weighted least squares
        //const S dist = std::sqrt(dx*dx+dy*dy);
        // we should really use radius to scale this weight!!!
        //const S weight = 1.f / (0.001f + dist);
        size_t idx = istart+i;

        // but for now, keep it simple
        xs[icnt++] = x[idx] - xt;
        xs[icnt++] = y[idx] - yt;
        vs[i] = u[idx];
    }

    // generate the least squares fit (not weighted yet)
    // template params are type, dimensions, polynomial order
    WLSPoly<S,2,1> lsfit;
    lsfit.solve(xs, vs);

    // evaluate at xt,yt, the origin
    std::vector<S> xep = {0.0, 0.0};
    return (S)lsfit.eval(xep);
}


//
// Approximate a spatial derivative from a number of irregularly-spaced points
//
template <class S, class A>
A least_squares_val(const S xt, const S yt, const S zt,
                    const Vector<S>& x, const Vector<S>& y,
                    const Vector<S>& z, const Vector<A>& u,
                    const size_t istart, const size_t iend) {

    //printf("  target point at %g %g %g\n", xt, yt, zt);
    S sn = 0.0f;
    S sx = 0.0f;
    S sy = 0.0f;
    S sz = 0.0f;
    S sx2 = 0.0f;
    S sy2 = 0.0f;
    S sz2 = 0.0f;
    S sv = 0.0f;
    S sxv = 0.0f;
    S syv = 0.0f;
    S szv = 0.0f;
    S sxy = 0.0f;
    S sxz = 0.0f;
    S syz = 0.0f;
    for (size_t i=istart; i<iend; ++i) {
        const S dx = x[i] - xt;
        const S dy = y[i] - yt;
        const S dz = z[i] - zt;
        const S dist = std::sqrt(dx*dx+dy*dy+dz*dz);
        //printf("    point %d at %g %g %g dist %g with value %g\n", i, x[i], y[i], z[i], u[i]);
        //printf("    point %d at %g %g %g dist %g with value %g\n", i, dx, dy, dz, dist, u[i]);
        const S weight = 1.f / (0.001f + dist);
        //const float oods = 1.0f / 
        //nsum
        // see https://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29
        // must solve a system of equations for ax + by + cz + d = 0
        // while minimizing the square error, this is a 4x4 matrix solve
        // ideally while also weighting the data points by their distance

        // compute sums of moments
        //const float weight = 1.f;
        sn += weight;
        sx += weight*dx;
        sy += weight*dy;
        sz += weight*dz;
        sv += weight*u[i];
        sxy += weight*dx*dy;
        sxz += weight*dx*dz;
        syz += weight*dy*dz;
        sxv += weight*dx*u[i];
        syv += weight*dy*u[i];
        szv += weight*dz*u[i];
        sx2 += weight*dx*dx;
        sy2 += weight*dy*dy;
        sz2 += weight*dz*dz;
    }
    // 47 flops per iter
    //printf("    sums are %g %g %g %g %g ...\n", sx, sy, sz, sv, sxy);

    // now begin to solve the equation
    const S i1 = sx/sxz - sn/sz;
    const S i2 = sx2/sxz - sx/sz;
    const S i3 = sxy/sxz - sy/sz;
    const S i4 = sxv/sxz - sv/sz;
    const S j1 = sy/syz - sn/sz;
    const S j2 = sxy/syz - sx/sz;
    const S j3 = sy2/syz - sy/sz;
    const S j4 = syv/syz - sv/sz;
    const S k1 = sz/sz2 - sn/sz;
    const S k2 = sxz/sz2 - sx/sz;
    const S k3 = syz/sz2 - sy/sz;
    const S k4 = szv/sz2 - sv/sz;
    const S q1 = i3*j1 - i1*j3;
    const S q2 = i3*j2 - i2*j3;
    const S q3 = i3*j4 - i4*j3;
    const S r1 = i3*k1 - i1*k3;
    const S r2 = i3*k2 - i2*k3;
    const S r3 = i3*k4 - i4*k3;
    // 18*3 = 54 flops

    const A b1 = (r2*q3 - r3*q2) / (r2*q1 - r1*q2);
    // 7 more
    //printf("    b1 is %g\n", b1);
    //const float b2 = r3/r2 - b1*r1/r2;
    //printf("    b2 is %g\n", b2);
    //const float b3 = j4/j3 - b1*j1/j3 - b2*j2/j3;
    //printf("    b3 is %g\n", b3);
    //const float b4 = sv/sz - b1/sz - b2*sx/sz - b3*sy/sz;
    //printf("    b4 is %g\n", b4);

    // when 16 contributing points, this is 813 flops

    //if (fabs(u[istart]) > 0.0) exit(0);
    return b1;
}

