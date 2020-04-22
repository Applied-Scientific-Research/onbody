/*
 * ongrav3d - testbed for an O(N) 3d gravitational solver
 *
 * Copyright (c) 2017-20, Mark J Stock
 */

#include <cstdlib>
#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>	// for sort and minmax
#include <numeric>	// for iota
#include <future>	// for async


#define STORE float
#define ACCUM double

const char* progname = "ongrav3d";

//
// The inner, scalar kernel
//
template <class S, class A>
static inline void nbody_kernel(const S sx, const S sy, const S sz,
                                const S sr, const S sm,
                                const S tx, const S ty, const S tz,
                                A& __restrict__ tax, A& __restrict__ tay, A& __restrict__ taz) {
    // 19 flops
    const S dx = sx - tx;
    const S dy = sy - ty;
    const S dz = sz - tz;
    S r2 = dx*dx + dy*dy + dz*dz + sr*sr;
    r2 = sm/(r2*std::sqrt(r2));
    tax += r2 * dx;
    tay += r2 * dy;
    taz += r2 * dz;
}

static inline int nbody_kernel_flops() { return 19; }

template <class S, class A, int PD, int SD, int OD> class Parts;

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    for (size_t j=jstart; j<jend; ++j) {
        nbody_kernel(srcs.x[0][j],  srcs.x[1][j], srcs.x[2][j], srcs.r[j], srcs.s[0][j],
                     targs.x[0][i], targs.x[1][i], targs.x[2][i],
                     targs.u[0][i], targs.u[1][i], targs.u[2][i]);
    }
}

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t istart, const size_t iend) {
    //printf("    compute srcs %ld-%ld on targs %ld-%ld\n", jstart, jend, istart, iend);
    for (size_t i=istart; i<iend; ++i) {
        for (size_t j=jstart; j<jend; ++j) {
            nbody_kernel(srcs.x[0][j],  srcs.x[1][j], srcs.x[2][j], srcs.r[j], srcs.s[0][j],
                         targs.x[0][i], targs.x[1][i], targs.x[2][i],
                         targs.u[0][i], targs.u[1][i], targs.u[2][i]);
    }
    }
}

template <class S, int PD, int SD> class Tree;

template <class S, class A, int PD, int SD, int OD>
void tpinter(const Tree<S,PD,SD>& __restrict__ stree, const size_t j,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    nbody_kernel(stree.x[0][j], stree.x[1][j], stree.x[2][j], stree.pr[j], stree.s[0][j],
                 targs.x[0][i], targs.x[1][i], targs.x[2][i],
                 targs.u[0][i], targs.u[1][i], targs.u[2][i]);
}

//
// Now we can include the tree-building and recursion code
//
#include "barneshut.h"
//
//

//
// Approximate a spatial derivative from a number of irregularly-spaced points
//
template <class S, class A>
A least_squares_val(const S xt, const S yt, const S zt,
                    const std::vector<S>& x, const std::vector<S>& y,
                    const std::vector<S>& z, const std::vector<A>& u,
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

//
// Data structure for accumulating interaction counts
//
struct fastsumm_stats {
    size_t sltl, sbtl, sltb, sbtb, tlc, lpc, bpc;
};

//
// Caller for the fast summation O(N) method
//
// ittn is the target tree node that this routine will work on
// itsv is the source tree node vector that will affect ittn
//
// We will change u,v,w for the targs points and the eqtargs equivalent points
//
template <class S, class A, int PD, int SD, int OD>
struct fastsumm_stats nbody_fastsumm(const Parts<S,A,PD,SD,OD>& srcs, const Parts<S,A,PD,SD,OD>& eqsrcs, const Tree<S,PD,SD>& stree,
                    Parts<S,A,PD,SD,OD>& targs, Parts<S,A,PD,SD,OD>& eqtargs, const Tree<S,PD,SD>& ttree,
                    const size_t ittn, std::vector<size_t> istv_in, const float theta) {

    // start counters
    struct fastsumm_stats stats = {0, 0, 0, 0, 0, 0, 0};

    // quit out if there are no particles in this box
    if (ttree.num[ittn] < 1) return stats;

    //printf("Targ box %d is affected by %lu source boxes at this level\n",ittn,istv.size());
    const bool targetIsLeaf = ttree.num[ittn] <= blockSize;

    // prepare the target arrays for accumulations
    if (targetIsLeaf) {
        stats.tlc++;
        // zero the velocities
        std::fill_n(&(targs.u[0][ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);
        std::fill_n(&(targs.u[1][ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);
        std::fill_n(&(targs.u[2][ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);

        if (ittn > 1) {
            // prolongation operation: take the parent's equiv points and move any
            // velocity from those to our real points
            const size_t destStart = ttree.ioffset[ittn];
            const size_t destNum = ttree.num[ittn];
            const size_t origStart = ttree.epoffset[ittn/2] + (blockSize/2) * (ittn%2);
            //const size_t origNum = (destNum+1)/2;
            //printf("  copying parent equiv parts %d to %d to our own real parts %d to %d\n",
            //       origStart, origStart+origNum, destStart, destStart+destNum);
            for (size_t i=0; i<destNum; ++i) {
                const size_t idest = destStart + i;
                const size_t iorig = origStart + i/2;
                //printf("    %d at %g %g %g is parent of %d at %g %g %g\n",
                //       iorig, eqtargs.x[iorig], eqtargs.y[iorig], eqtargs.z[iorig],
                //       idest,   targs.x[idest],   targs.y[idest],   targs.z[idest]);
                // second take, use linear least squares to approximate value
                if (true) {
                    const size_t nearest = 16;
                    const size_t istart = nearest*(iorig/nearest);
                    const size_t iend = istart+nearest;
                    //printf("  approximating velocity at equiv pt %d from equiv pt %d\n", idest, iorig);
                    targs.u[0][idest] = least_squares_val(targs.x[0][idest], targs.x[1][idest], targs.x[2][idest],
                                                          eqtargs.x[0], eqtargs.x[1], eqtargs.x[1],
                                                          eqtargs.u[0], istart, iend);
                    targs.u[1][idest] = least_squares_val(targs.x[0][idest], targs.x[1][idest], targs.x[2][idest],
                                                          eqtargs.x[0], eqtargs.x[1], eqtargs.x[2],
                                                          eqtargs.u[1], istart, iend);
                    targs.u[2][idest] = least_squares_val(targs.x[0][idest], targs.x[1][idest], targs.x[2][idest],
                                                          eqtargs.x[0], eqtargs.x[1], eqtargs.x[2],
                                                          eqtargs.u[2], istart, iend);
                } else {
                    // as a first take, simply copy the result to the children
                    targs.u[0][idest] = eqtargs.u[0][iorig];
                    targs.u[1][idest] = eqtargs.u[1][iorig];
                    targs.u[2][idest] = eqtargs.u[2][iorig];
                }
            }
            stats.lpc++;
        }

    } else {
        // zero the equivalent particle velocities
        std::fill_n(&(eqtargs.u[0][ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);
        std::fill_n(&(eqtargs.u[1][ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);
        std::fill_n(&(eqtargs.u[2][ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);

        if (ittn > 1) {
            // prolongation operation: take the parent's equiv points and move any
            // velocity from those to our equiv points
            const size_t destStart = ttree.epoffset[ittn];
            const size_t destNum = ttree.epnum[ittn];
            const size_t origStart = ttree.epoffset[ittn/2] + (blockSize/2) * (ittn%2);
            //const size_t origNum = (destNum+1)/2;
            //printf("  copying parent equiv parts %d to %d to our own equiv parts %d to %d\n",
            //       origStart, origStart+origNum, destStart, destStart+destNum);

            //for (size_t i=0; i<(ttree.epnum[ittn]+1)/2; ++i) {
            //    size_t ipe = ttree.epoffset[ittn]/2 + i;
            //    printf("    %d  %g %g %g\n", ipe, eqtargs.u[ipe], eqtargs.v[ipe], eqtargs.w[ipe]);
            //}

            for (size_t i=0; i<destNum; ++i) {
                const size_t idest = destStart + i;
                const size_t iorig = origStart + i/2;
                //printf("    %d at %g %g %g is parent of %d at %g %g %g\n",
                //       iorig, eqtargs.x[iorig], eqtargs.y[iorig], eqtargs.z[iorig],
                //       idest, eqtargs.x[idest], eqtargs.y[idest], eqtargs.z[idest]);
                // second take, apply gradient of value to delta location
                if (true) {
                    const size_t nearest = 16;
                    const size_t istart = nearest*(iorig/nearest);
                    const size_t iend = istart+nearest;
                    //printf("  approximating velocity at equiv pt %d from equiv pt %d\n", idest, iorig);
                    eqtargs.u[0][idest] = least_squares_val(eqtargs.x[0][idest], eqtargs.x[1][idest], eqtargs.x[2][idest],
                                                            eqtargs.x[0], eqtargs.x[1], eqtargs.x[2],
                                                            eqtargs.u[0], istart, iend);
                    eqtargs.u[1][idest] = least_squares_val(eqtargs.x[0][idest], eqtargs.x[1][idest], eqtargs.x[2][idest],
                                                            eqtargs.x[0], eqtargs.x[1], eqtargs.x[2],
                                                            eqtargs.u[1], istart, iend);
                    eqtargs.u[2][idest] = least_squares_val(eqtargs.x[0][idest], eqtargs.x[1][idest], eqtargs.x[2][idest],
                                                            eqtargs.x[0], eqtargs.x[1], eqtargs.x[2],
                                                            eqtargs.u[2], istart, iend);
                } else {
                    // as a first take, simply copy the result to the children
                    eqtargs.u[0][idest] = eqtargs.u[0][iorig];
                    eqtargs.u[1][idest] = eqtargs.u[1][iorig];
                    eqtargs.u[2][idest] = eqtargs.u[2][iorig];
                }
            }
            stats.bpc++;
        }
    }

    // initialize a new vector of source boxes to pass to this target box's children
    std::vector<size_t> cstv;

    // make a local copy of the input source tree vector
    std::vector<size_t> istv = istv_in;

    // for target box ittn, check all unaccounted-for source boxes
    size_t num_istv = istv.size();
    for (size_t i=0; i<num_istv; i++) {
        const size_t sn = istv[i];

        // skip this loop iteration
        if (stree.num[sn] < 1) continue;

        const bool sourceIsLeaf = stree.num[sn] <= blockSize;
        //printf("  source %d affects target %d\n",sn,ittn);

        // if source box is a leaf node, just compute the influence and return?
        // this assumes target box is also a leaf node!
        if (sourceIsLeaf and targetIsLeaf) {
            //printf("    real on real, srcs %d to %d, targs %d to %d\n", stree.ioffset[sn], stree.ioffset[sn]   + stree.num[sn], ttree.ioffset[ittn], ttree.ioffset[ittn] + ttree.num[ittn]);

            // compute all-on-all direct influence
            for (size_t i = ttree.ioffset[ittn]; i < ttree.ioffset[ittn] + ttree.num[ittn]; i++) {
            for (size_t j = stree.ioffset[sn];   j < stree.ioffset[sn]   + stree.num[sn];   j++) {
                nbody_kernel(srcs.x[0][j],  srcs.x[1][j],  srcs.x[2][j], srcs.r[j], srcs.s[0][j],
                             targs.x[0][i], targs.x[1][i], targs.x[2][i],
                             targs.u[0][i], targs.u[1][i], targs.u[2][i]);
            }
            }
            stats.sltl++;
            continue;
        }

        // distance from box center of mass to target point
        S dist = 0.0;
        for (int d=0; d<PD; ++d) dist += std::pow(stree.x[d][sn] - ttree.x[d][ittn], 2);
        dist = std::sqrt(dist);
        const S diag = stree.nr[sn] + ttree.nr[ittn];
        //printf("  src box %d is %g away and diag %g\n",sn, dist, diag);

        // split on what to do with this pair
        if (dist / diag > theta) {
            // it is far enough - we can approximate
            //printf("    well-separated\n");

            if (sourceIsLeaf) {
                // compute real source particles on equivalent target points
                for (size_t i = ttree.epoffset[ittn]; i < ttree.epoffset[ittn] + ttree.epnum[ittn]; i++) {
                for (size_t j = stree.ioffset[sn];    j < stree.ioffset[sn]    + stree.num[sn];     j++) {
                    nbody_kernel(srcs.x[0][j],    srcs.x[1][j],    srcs.x[2][j], srcs.r[j], srcs.s[0][j],
                                 eqtargs.x[0][i], eqtargs.x[1][i], eqtargs.x[2][i],
                                 eqtargs.u[0][i], eqtargs.u[1][i], eqtargs.u[2][i]);
                }
                }
                stats.sltb++;

            } else if (targetIsLeaf) {
                // compute equivalent source particles on real target points
                for (size_t i = ttree.ioffset[ittn]; i < ttree.ioffset[ittn] + ttree.num[ittn]; i++) {
                for (size_t j = stree.epoffset[sn];  j < stree.epoffset[sn]  + stree.epnum[sn]; j++) {
                    nbody_kernel(eqsrcs.x[0][j], eqsrcs.x[1][j], eqsrcs.x[2][j], eqsrcs.r[j], eqsrcs.s[0][j],
                                 targs.x[0][i],  targs.x[1][i],  targs.x[2][i],
                                 targs.u[0][i],  targs.u[1][i],  targs.u[2][i]);
                }
                }
                stats.sbtl++;

            } else {
                // compute equivalent source particles on equivalent target points
                for (size_t i = ttree.epoffset[ittn]; i < ttree.epoffset[ittn] + ttree.epnum[ittn]; i++) {
                for (size_t j = stree.epoffset[sn];   j < stree.epoffset[sn]   + stree.epnum[sn];   j++) {
                    nbody_kernel(eqsrcs.x[0][j],  eqsrcs.x[1][j],  eqsrcs.x[2][j], eqsrcs.r[j], eqsrcs.s[0][j],
                                 eqtargs.x[0][i], eqtargs.x[1][i], eqtargs.x[2][i],
                                 eqtargs.u[0][i], eqtargs.u[1][i], eqtargs.u[2][i]);
                }
                }
                stats.sbtb++;
            }

        } else if (ttree.nr[ittn] > 0.7*stree.nr[sn]) {
        //} else if (true) {
            // target box is larger than source box; try to refine targets first
            //printf("    not well-separated, target is larger\n");

            if (targetIsLeaf) {
                // this means source is NOT leaf
                // put children of source box onto the end of the current list
                istv.push_back(2*sn);
                istv.push_back(2*sn+1);
                num_istv += 2;
                //printf("    pushing %d and %d to the end of this list\n", 2*sn, 2*sn+1);
            } else {
                // put this source box on the new list for target's children
                cstv.push_back(sn);
                //printf("    pushing %d to the end of the new list\n", sn);
            }

        } else {
            // source box is larger than target box; try to refine sources first
            //printf("    not well-separated, source is larger\n");

            if (sourceIsLeaf) {
                // this means target is NOT leaf
                // put this source box on the new list for target's children
                cstv.push_back(sn);
                //printf("    pushing %d to the end of the new list\n", sn);
            } else {
                // put children of source box onto the end of the current list
                istv.push_back(2*sn);
                istv.push_back(2*sn+1);
                num_istv += 2;
                //printf("    pushing %d and %d to the end of this list\n", 2*sn, 2*sn+1);
            }
        }
        //printf("    istv now has %lu entries\n",istv.size());
    }

    if (targetIsLeaf) {
        //printf("  leaf box %ld  sltl %ld  sbtl %ld\n", ittn, stats.sltl, stats.sbtl);

    } else {
        //printf("  non-leaf box %ld                     sltb %ld  sbtb %ld\n", ittn, stats.sltb, stats.sbtb);
        // prolongation of equivalent particle velocities to children's equivalent particles

        // recurse onto the target box's children
        struct fastsumm_stats cstats1, cstats2;

        #pragma omp task shared(srcs,eqsrcs,stree,targs,eqtargs,ttree,cstats1)
        cstats1 = nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree, 2*ittn, cstv, theta);

        #pragma omp task shared(srcs,eqsrcs,stree,targs,eqtargs,ttree,cstats2)
        cstats2 = nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree, 2*ittn+1, cstv, theta);

        // accumulate the child box's stats - but must wait until preceding tasks complete
        #pragma omp taskwait
        stats.sltl += cstats1.sltl + cstats2.sltl;
        stats.sbtl += cstats1.sbtl + cstats2.sbtl;
        stats.sltb += cstats1.sltb + cstats2.sltb;
        stats.sbtb += cstats1.sbtb + cstats2.sbtb;
        stats.tlc  += cstats1.tlc  + cstats2.tlc;
        stats.lpc  += cstats1.lpc  + cstats2.lpc;
        stats.bpc  += cstats1.bpc  + cstats2.bpc;
    }

    // report counter results
    if (ittn == 1) {
        #pragma omp taskwait
        printf("  %ld target leaf nodes averaged %g leaf-leaf and %g equiv-leaf interactions\n",
               stats.tlc, stats.sltl/(float)stats.tlc, stats.sbtl/(float)stats.tlc);
        printf("  sltl %ld  sbtl %ld  sltb %ld  sbtb %ld\n", stats.sltl, stats.sbtl, stats.sltb, stats.sbtb);
        printf("  leaf prolongation count %ld  box pc %ld\n", stats.lpc, stats.bpc);
    }

    //printf("  box %ld  sltl %ld  sbtl %ld  sltb %ld  sbtb %ld\n", ittn, stats.sltl, stats.sbtl, stats.sltb, stats.sbtb);
    return stats;
}

//
// basic usage
//
static void usage() {
    fprintf(stderr, "Usage: %s [-n=<nparticles>]\n", progname);
    exit(1);
}

//
// main routine - run the program
//
int main(int argc, char *argv[]) {

    static std::vector<int> test_iterations = {1, 1, 1, 1, 0};
    bool just_build_trees = false;
    size_t numSrcs = 10000;
    size_t numTargs = 10000;
    float theta = 4.0;
    std::vector<double> treetime(test_iterations.size(), 0.0);

    for (int i=1; i<argc; i++) {
        if (strncmp(argv[i], "-n=", 3) == 0) {
            size_t num = atoi(argv[i]+3);
            if (num < 1) usage();
            numSrcs = num;
            numTargs = num;
        } else if (strncmp(argv[i], "-t=", 3) == 0) {
            float testtheta = atof(argv[i]+3);
            if (testtheta < 0.0001) usage();
            theta = testtheta;
        }
    }

    printf("Running %s with %ld sources and %ld targets\n", progname, numSrcs, numTargs);
    printf("  block size of %ld and theta %g\n\n", blockSize, theta);

    // if problem is too big, skip some number of target particles
    //size_t ntskip = std::max(1, (int)((float)numSrcs*(float)numTargs/2.e+9));
    size_t ntskip = std::max(1, (int)((float)numTargs/1.e+4));

    printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<STORE,ACCUM,3,1,3> srcs(numSrcs);
    // initialize particle data
    srcs.random_in_cube();
    for (auto& m : srcs.s[0]) { m = std::abs(m); }

    Parts<STORE,ACCUM,3,1,3> targs(numTargs);
    // initialize particle data
    targs.random_in_cube();
    //for (auto& m : targs.m) { m = 1.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // initialize and generate tree
    printf("\nBuilding the source tree\n");
    printf("  with %ld particles and block size of %ld\n", numSrcs, blockSize);
    start = std::chrono::system_clock::now();
    Tree<STORE,3,1> stree(0);
    // split this node and recurse
    (void) makeTree(srcs, stree);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());
    treetime[1] += elapsed_seconds.count();
    treetime[2] += elapsed_seconds.count();
    treetime[3] += elapsed_seconds.count();
    treetime[4] += elapsed_seconds.count();

    // find equivalent particles
    printf("\nCalculating equivalent particles\n");
    start = std::chrono::system_clock::now();
    Parts<STORE,ACCUM,3,1,3> eqsrcs((stree.numnodes/2) * blockSize);
    printf("  need %ld particles\n", eqsrcs.n);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  allocate eqsrcs structures:\t[%.4f] seconds\n", elapsed_seconds.count());
    treetime[2] += elapsed_seconds.count();
    treetime[3] += elapsed_seconds.count();
    treetime[4] += elapsed_seconds.count();

    // first, reorder tree until all parts are adjacent in space-filling curve
    start = std::chrono::system_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) refineTree(srcs, stree, 1);
    #pragma omp taskwait
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  refine within leaf nodes:\t[%.4f] seconds\n", elapsed_seconds.count());
    treetime[2] += elapsed_seconds.count();
    treetime[3] += elapsed_seconds.count();
    treetime[4] += elapsed_seconds.count();
    //for (size_t i=0; i<stree.num[1]; ++i)
    //    printf("%d %g %g %g\n", i, srcs.x[i], srcs.y[i], srcs.z[i]);

    // then, march through arrays merging pairs as you go up
    start = std::chrono::system_clock::now();
    (void) calcEquivalents(srcs, eqsrcs, stree, 1);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  create equivalent parts:\t[%.4f] seconds\n", elapsed_seconds.count());
    treetime[2] += elapsed_seconds.count();
    treetime[3] += elapsed_seconds.count();
    treetime[4] += elapsed_seconds.count();


    // don't need the target tree for treecode, but will for fast code
    Tree<STORE,3,1> ttree(0);
    if (test_iterations[3] > 0 or test_iterations[4] > 0) {
        printf("\nBuilding the target tree\n");
        printf("  with %ld particles and block size of %ld\n", numTargs, blockSize);
        start = std::chrono::system_clock::now();
        // split this node and recurse
        (void) makeTree(targs, ttree);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());
        treetime[3] += elapsed_seconds.count();
        treetime[4] += elapsed_seconds.count();
    }

    // find equivalent points
    Parts<STORE,ACCUM,3,1,3> eqtargs(0);
    if (test_iterations[4] > 0) {
        printf("\nCalculating equivalent targ points\n");
        start = std::chrono::system_clock::now();
        eqtargs = Parts<STORE,ACCUM,3,1,3>((ttree.numnodes/2) * blockSize);
        printf("  need %ld particles\n", eqtargs.n);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        printf("  allocate eqtargs structures:\t[%.4f] seconds\n", elapsed_seconds.count());
        treetime[4] += elapsed_seconds.count();

        // first, reorder tree until all parts are adjacent in space-filling curve
        start = std::chrono::system_clock::now();
        #pragma omp parallel
        #pragma omp single
        (void) refineTree(targs, ttree, 1);
        #pragma omp taskwait
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        printf("  refine within leaf nodes:\t[%.4f] seconds\n", elapsed_seconds.count());
        treetime[4] += elapsed_seconds.count();

        // then, march through arrays merging pairs as you go up
        start = std::chrono::system_clock::now();
        (void) calcEquivalents(targs, eqtargs, ttree, 1);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        printf("  create equivalent parts:\t[%.4f] seconds\n", elapsed_seconds.count());
        treetime[4] += elapsed_seconds.count();
    }

    if (just_build_trees) exit(0);

    //
    // Run the O(N^2) implementation
    //
    printf("\nRun the naive O(N^2) method (every %ld particles)\n", ntskip);
    double minNaive = 1e30;
    float flops = 0.0;
    for (int i = 0; i < test_iterations[0]; ++i) {
        targs.zero_vels();
        start = std::chrono::system_clock::now();
        flops = nbody_naive(srcs, targs, ntskip);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minNaive = std::min(minNaive, dt);
    }
    printf("[onbody naive]:\t\t\t[%.4f] seconds\n", minNaive * (float)ntskip);
    printf("  GFlop: %.2f and GFlop/s: %.3f\n", flops*1.e-9*(float)ntskip, flops*1.e-9/minNaive);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    std::vector<ACCUM> naiveu(targs.u[0].begin(), targs.u[0].end());

    ACCUM errsum = 0.0;
    ACCUM errcnt = 0.0;
    ACCUM maxerr = 0.0;

    //
    // Run a simple O(NlogN) treecode - boxes approximate as particles
    //
    if (test_iterations[1] > 0) {
    printf("\nRun the treecode O(NlogN)\n");
    double minTreecode = 1e30;
    for (int i = 0; i < test_iterations[1]; ++i) {
        targs.zero_vels();
        start = std::chrono::system_clock::now();
        flops = nbody_treecode1(srcs, stree, targs, theta);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minTreecode = std::min(minTreecode, dt);
    }
    printf("[onbody treecode]:\t\t[%.4f] seconds\n", minTreecode);
    printf("  GFlop: %.3f and GFlop/s: %.3f\n", flops*1.e-9, flops*1.e-9/minTreecode);
    printf("[treecode total]:\t\t[%.4f] seconds\n", treetime[1] + minTreecode);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    // save the results for comparison
    std::vector<ACCUM> treecodeu(targs.u[0].begin(), targs.u[0].end());

    // compare accuracy
    errsum = 0.0; errcnt = 0.0; maxerr = 0.0;
    for (size_t i=0; i<numTargs; i+=ntskip) {
        ACCUM thiserr = treecodeu[i]-naiveu[i];
        errsum += thiserr*thiserr;
        if (thiserr*thiserr > maxerr) maxerr = thiserr*thiserr;
        errcnt += naiveu[i]*naiveu[i];
    }
    printf("error in treecode (RMS/max):\t%g / %g\n", std::sqrt(errsum/errcnt), std::sqrt(maxerr/(ntskip*errcnt/(float)numTargs)));
    }


    //
    // Run a better O(NlogN) treecode - boxes use equivalent particles
    //
    if (test_iterations[2] > 0) {
    printf("\nRun the treecode O(NlogN) with equivalent particles\n");
    double minTreecode2 = 1e30;
    for (int i = 0; i < test_iterations[2]; ++i) {
        targs.zero_vels();
        start = std::chrono::system_clock::now();
        flops = nbody_treecode2(srcs, eqsrcs, stree, targs, theta);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minTreecode2 = std::min(minTreecode2, dt);
    }
    printf("[onbody treecode2]:\t\t[%.4f] seconds\n", minTreecode2);
    printf("  GFlop: %.3f and GFlop/s: %.3f\n", flops*1.e-9, flops*1.e-9/minTreecode2);
    printf("[treecode2 total]:\t\t[%.4f] seconds\n", treetime[2] + minTreecode2);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    // save the results for comparison
    std::vector<ACCUM> treecodeu2(targs.u[0].begin(), targs.u[0].end());

    // compare accuracy
    errsum = 0.0; errcnt = 0.0; maxerr = 0.0;
    for (size_t i=0; i<numTargs; i+=ntskip) {
        ACCUM thiserr = treecodeu2[i]-naiveu[i];
        errsum += thiserr*thiserr;
        if (thiserr*thiserr > maxerr) maxerr = thiserr*thiserr;
        errcnt += naiveu[i]*naiveu[i];
    }
    printf("error in treecode2 (RMS/max):\t%g / %g\n", std::sqrt(errsum/errcnt), std::sqrt(maxerr/(ntskip*errcnt/(float)numTargs)));
    }


    //
    // Run a better O(NlogN) treecode - boxes use equivalent particles - lists are boxwise
    //
    if (test_iterations[3] > 0) {
    printf("\nRun the treecode O(NlogN) with equivalent particles and boxwise interactions\n");
    double minTreecode3 = 1e30;
    for (int i = 0; i < test_iterations[3]; ++i) {
        targs.zero_vels();
        start = std::chrono::system_clock::now();
        flops = nbody_treecode3(srcs, eqsrcs, stree, targs, ttree, theta);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minTreecode3 = std::min(minTreecode3, dt);
    }
    printf("[onbody treecode3]:\t\t[%.4f] seconds\n", minTreecode3);
    printf("  GFlop: %.3f and GFlop/s: %.3f\n", flops*1.e-9, flops*1.e-9/minTreecode3);
    printf("[treecode3 total]:\t\t[%.4f] seconds\n", treetime[3] + minTreecode3);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    // save the results for comparison
    std::vector<ACCUM> treecodeu3(targs.u[0].begin(), targs.u[0].end());

    // compare accuracy
    errsum = 0.0; errcnt = 0.0; maxerr = 0.0;
    for (size_t i=0; i<numTargs; i+=ntskip) {
        ACCUM thiserr = treecodeu3[i]-naiveu[i];
        errsum += thiserr*thiserr;
        if (thiserr*thiserr > maxerr) maxerr = thiserr*thiserr;
        errcnt += naiveu[i]*naiveu[i];
    }
    printf("error in treecode3 (RMS/max):\t%g / %g\n", std::sqrt(errsum/errcnt), std::sqrt(maxerr/(ntskip*errcnt/(float)numTargs)));
    }


    //
    // Run the new O(N) equivalent particle method
    //
    if (test_iterations[4] > 0) {
    printf("\nRun the fast O(N) method\n");
    double minFast = 1e30;
    for (int i = 0; i < test_iterations[3]; ++i) {
        targs.zero_vels();
        start = std::chrono::system_clock::now();
        std::vector<size_t> source_boxes = {1};
        // theta=0.82f roughly matches treecode2's 1.4f re: num of leaf-leaf interactions
        // theta=1.5f roughly matches treecode2's 1.4f re: RMS error
        #pragma omp parallel
        #pragma omp single
        (void) nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree,
                              1, source_boxes, theta);
        #pragma omp taskwait
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minFast = std::min(minFast, dt);
    }
    printf("[onbody fast]:\t\t\t[%.4f] seconds\n", minFast);
    printf("[fast total]:\t\t\t[%.4f] seconds\n", treetime[4] + minFast);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    // save the results for comparison
    std::vector<ACCUM> fastu(targs.u[0].begin(), targs.u[0].end());

    // compare accuracy
    errsum = 0.0; errcnt = 0.0; maxerr = 0.0;
    for (size_t i=0; i<numTargs; i+=ntskip) {
        ACCUM thiserr = fastu[i]-naiveu[i];
        errsum += thiserr*thiserr;
        if (thiserr*thiserr > maxerr) maxerr = thiserr*thiserr;
        errcnt += naiveu[i]*naiveu[i];
    }
    printf("error in fastsumm (RMS/max):\t%g / %g\n", std::sqrt(errsum/errcnt), std::sqrt(maxerr/(ntskip*errcnt/(float)numTargs)));
    }

    printf("\nDone.\n");

    return 0;
}
