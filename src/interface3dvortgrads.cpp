/*
 * interface3dvortgrads.cpp - interface code for Barnes-Hut treecode
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 */

#include <cstdlib>
#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <chrono>


//#define USE_RM_KERNEL
#define USE_EXPONENTIAL_KERNEL


#ifdef USE_RM_KERNEL
template <class S>
static inline void core_func (const S distsq, const S sr,
                              S* const __restrict__ r3, S* const __restrict__ bbb) {
  const S r2 = distsq + sr*sr;
  *r3 = S(1.0) / (r2*std::sqrt(r2));
  *bbb = S(-3.0) * (*r3) / r2;
}
int flops_tp_grads () { return 7; }
#endif

#ifdef USE_EXPONENTIAL_KERNEL
template <class S>
static inline void core_func (const S distsq, const S sr,
                              S* const __restrict__ r3, S* const __restrict__ bbb) {
  const S dist = std::sqrt(distsq);
  const S corefac = S(1.0) / std::pow(sr,3);
  const S d3 = distsq * dist;
  const S reld3 = d3 * corefac;
  // 6 flops to here
  if (reld3 > S(16.0)) {
    *r3 = S(1.0) / d3;
    *bbb = S(-3.0) / (d3 * distsq);
    // this is 4 flops and is most likely
  } else if (reld3 < S(0.001)) {
    *r3 = corefac;
    *bbb = S(-1.5) * dist * corefac * corefac;
    // this is 5 flops
  } else {
    const S expreld3 = std::exp(-reld3);
    *r3 = (S(1.0) - expreld3) / d3;
    *bbb = S(3.0) * (corefac*expreld3 - *r3) / distsq;
    // this is 9 flops
  }
}
int flops_tp_grads () { return 11; }
#endif


//
// The inner, scalar kernel
//
template <class S, class A>
static inline void nbody_kernel(const S sx, const S sy, const S sz,
                                const S ssx, const S ssy, const S ssz, const S sr,
                                const S tx, const S ty, const S tz,
                                A& __restrict__ tu, A& __restrict__ tv, A& __restrict__ tw,
                                A& __restrict__ tux, A& __restrict__ tvx, A& __restrict__ twx,
                                A& __restrict__ tuy, A& __restrict__ tvy, A& __restrict__ twy,
                                A& __restrict__ tuz, A& __restrict__ tvz, A& __restrict__ twz) {
    // 56 flops
    const S dx = tx - sx;
    const S dy = ty - sy;
    const S dz = tz - sz;
    //const S r2 = dx*dx + dy*dy + dz*dz + sr*sr;
    //const S r3 = (S)1.0 / (r2*std::sqrt(r2));
    //const S bbb = S(-3.0) * r3 / r2;
    S r3, bbb;
    (void) core_func<S>(dx*dx + dy*dy + dz*dz, sr, &r3, &bbb);
    S dxxw = dz*ssy - dy*ssz;
    S dyxw = dx*ssz - dz*ssx;
    S dzxw = dy*ssx - dx*ssy;
    tu += r3 * dxxw;
    tv += r3 * dyxw;
    tw += r3 * dzxw;
    dxxw *= bbb;
    dyxw *= bbb;
    dzxw *= bbb;
    tux += dx*dxxw;
    tvx += dx*dyxw + ssz*r3;
    twx += dx*dzxw - ssy*r3;
    tuy += dy*dxxw - ssz*r3;
    tvy += dy*dyxw;
    twy += dy*dzxw + ssx*r3;
    tuz += dz*dxxw + ssy*r3;
    tvz += dz*dyxw - ssx*r3;
    twz += dz*dzxw;
}

static inline int nbody_kernel_flops() { return 56 + flops_tp_grads(); }

template <class S, class A, int PD, int SD, int OD> class Parts;

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    for (size_t j=jstart; j<jend; ++j) {
        nbody_kernel(srcs.x[0][j],  srcs.x[1][j],  srcs.x[2][j],
                     srcs.s[0][j],  srcs.s[1][j],  srcs.s[2][j],
                     srcs.r[j],
                     targs.x[0][i], targs.x[1][i], targs.x[2][i],
                     targs.u[0][i], targs.u[1][i], targs.u[2][i],
                     targs.u[3][i], targs.u[4][i], targs.u[5][i],
                     targs.u[6][i], targs.u[7][i], targs.u[8][i],
                     targs.u[9][i], targs.u[10][i],targs.u[11][i]);
    }
}

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t istart, const size_t iend) {
    //printf("    compute srcs %ld-%ld on targs %ld-%ld\n", jstart, jend, istart, iend);
    for (size_t i=istart; i<iend; ++i) {
        for (size_t j=jstart; j<jend; ++j) {
            nbody_kernel(srcs.x[0][j],  srcs.x[1][j],  srcs.x[2][j],
                     srcs.s[0][j],  srcs.s[1][j],  srcs.s[2][j],
                     srcs.r[j],
                     targs.x[0][i], targs.x[1][i], targs.x[2][i],
                     targs.u[0][i], targs.u[1][i], targs.u[2][i],
                     targs.u[3][i], targs.u[4][i], targs.u[5][i],
                     targs.u[6][i], targs.u[7][i], targs.u[8][i],
                     targs.u[9][i], targs.u[10][i],targs.u[11][i]);
    }
    }
}

template <class S, int PD, int SD> class Tree;

template <class S, class A, int PD, int SD, int OD>
void tpinter(const Tree<S,PD,SD>& __restrict__ stree, const size_t j,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    nbody_kernel(stree.x[0][j], stree.x[1][j], stree.x[2][j],
                     stree.s[0][j], stree.s[1][j], stree.s[2][j],
                     stree.pr[j],
                     targs.x[0][i], targs.x[1][i], targs.x[2][i],
                     targs.u[0][i], targs.u[1][i], targs.u[2][i],
                     targs.u[3][i], targs.u[4][i], targs.u[5][i],
                     targs.u[6][i], targs.u[7][i], targs.u[8][i],
                     targs.u[9][i], targs.u[10][i],targs.u[11][i]);
}

//
// Now we can include the tree-building and recursion code
//
#include "barneshut.h"
//
//


//
// call this function from an external program
//
extern "C" float external_vel_solver_f_ (const int* nsrc,
                                         const float* sx, const float* sy, const float* sz,
                                         const float* ssx, const float* ssy, const float* ssz,
                                         const float* sr,
                                         const int* ntarg,
                                         const float* tx, const float* ty, const float* tz,
                                         float* tu,  float* tv,  float* tw,
                                         float* tux, float* tvx, float* twx,
                                         float* tuy, float* tvy, float* twy,
                                         float* tuz, float* tvz, float* twz) {
    float flops = 0.0;
    const bool silent = false;
    const bool createTargTree = true;
    const bool blockwise = true;

    if (!silent) printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double,3,3,12> srcs(*nsrc,true);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[2][i] = sz[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[0][i] = ssx[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[1][i] = ssy[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[2][i] = ssz[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<float,double,3,3,12> targs(*ntarg,false);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
    for (int i=0; i<*ntarg; ++i) targs.x[2][i] = tz[i];
    for (auto& m : targs.s[0]) { m = 1.0f; }
    for (auto& u : targs.u[0]) { u = 0.0f; }
    for (auto& u : targs.u[1]) { u = 0.0f; }
    for (auto& u : targs.u[2]) { u = 0.0f; }
    for (auto& u : targs.u[3]) { u = 0.0f; }
    for (auto& u : targs.u[4]) { u = 0.0f; }
    for (auto& u : targs.u[5]) { u = 0.0f; }
    for (auto& u : targs.u[6]) { u = 0.0f; }
    for (auto& u : targs.u[7]) { u = 0.0f; }
    for (auto& u : targs.u[8]) { u = 0.0f; }
    for (auto& u : targs.u[9]) { u = 0.0f; }
    for (auto& u : targs.u[10]) { u = 0.0f; }
    for (auto& u : targs.u[11]) { u = 0.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    if (!silent) printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // initialize and generate tree
    if (!silent) printf("\nBuilding the source tree\n");
    if (!silent) printf("  with %d particles and block size of %ld\n", *nsrc, blockSize);
    start = std::chrono::system_clock::now();
    Tree<float,3,3> stree(0);
    // split this node and recurse
    (void) makeTree(srcs, stree);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());

    // find equivalent particles
    if (!silent) printf("\nCalculating equivalent particles\n");
    start = std::chrono::system_clock::now();
    Parts<float,double,3,3,12> eqsrcs((stree.numnodes/2) * blockSize, true);
    if (!silent) printf("  need %ld particles\n", eqsrcs.n);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  allocate eqsrcs structures:\t[%.4f] seconds\n", elapsed_seconds.count());

    // reorder tree until all parts are adjacent in space-filling curve
    start = std::chrono::system_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) refineTree(srcs, stree, 1);
    #pragma omp taskwait
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  refine within leaf nodes:\t[%.4f] seconds\n", elapsed_seconds.count());

    // then, march through arrays merging pairs as you go up
    start = std::chrono::system_clock::now();
    (void) calcEquivalents(srcs, eqsrcs, stree, 1);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  create equivalent parts:\t[%.4f] seconds\n", elapsed_seconds.count());


    Tree<float,3,3> ttree(0);
    if (createTargTree or blockwise) {
        if (!silent) printf("\nBuilding the target tree\n");
        if (!silent) printf("  with %d particles and block size of %ld\n", *ntarg, blockSize);

        start = std::chrono::system_clock::now();
        // split this node and recurse
        (void) makeTree(targs, ttree);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        if (!silent) printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());
    }


    //
    // Run a better O(NlogN) treecode - boxes use equivalent particles
    //
    if (!silent) printf("\nRun the treecode O(NlogN) with equivalent particles\n");
    start = std::chrono::system_clock::now();

    if (blockwise) {
        flops += nbody_treecode3(srcs, eqsrcs, stree, targs, ttree, 2.5f);
    } else {
        flops += nbody_treecode2(srcs, eqsrcs, stree, targs, 2.5f);
    }

    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    double dt = elapsed_seconds.count();
    if (!silent) printf("  treecode summations:\t\t[%.4f] seconds\n\n", dt);


    // pull results from the object
    if (createTargTree) {
        // need to rearrange the results back in original order
        for (int i=0; i<*ntarg; ++i) tu[targs.gidx[i]] += targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[targs.gidx[i]] += targs.u[1][i];
        for (int i=0; i<*ntarg; ++i) tw[targs.gidx[i]] += targs.u[2][i];
        for (int i=0; i<*ntarg; ++i) tux[targs.gidx[i]] += targs.u[3][i];
        for (int i=0; i<*ntarg; ++i) tvx[targs.gidx[i]] += targs.u[4][i];
        for (int i=0; i<*ntarg; ++i) twx[targs.gidx[i]] += targs.u[5][i];
        for (int i=0; i<*ntarg; ++i) tuy[targs.gidx[i]] += targs.u[6][i];
        for (int i=0; i<*ntarg; ++i) tvy[targs.gidx[i]] += targs.u[7][i];
        for (int i=0; i<*ntarg; ++i) twy[targs.gidx[i]] += targs.u[8][i];
        for (int i=0; i<*ntarg; ++i) tuz[targs.gidx[i]] += targs.u[9][i];
        for (int i=0; i<*ntarg; ++i) tvz[targs.gidx[i]] += targs.u[10][i];
        for (int i=0; i<*ntarg; ++i) twz[targs.gidx[i]] += targs.u[11][i];
        //for (int i=0; i<*ntarg; ++i) printf("  %d %ld  %g %g\n", i, targs.gidx[i], tu[i], tv[i]);
    } else {
        // pull them out directly
        for (int i=0; i<*ntarg; ++i) tu[i] += targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[i] += targs.u[1][i];
        for (int i=0; i<*ntarg; ++i) tw[i] += targs.u[2][i];
        for (int i=0; i<*ntarg; ++i) tux[i] += targs.u[3][i];
        for (int i=0; i<*ntarg; ++i) tvx[i] += targs.u[4][i];
        for (int i=0; i<*ntarg; ++i) twx[i] += targs.u[5][i];
        for (int i=0; i<*ntarg; ++i) tuy[i] += targs.u[6][i];
        for (int i=0; i<*ntarg; ++i) tvy[i] += targs.u[7][i];
        for (int i=0; i<*ntarg; ++i) twy[i] += targs.u[8][i];
        for (int i=0; i<*ntarg; ++i) tuz[i] += targs.u[9][i];
        for (int i=0; i<*ntarg; ++i) tvz[i] += targs.u[10][i];
        for (int i=0; i<*ntarg; ++i) twz[i] += targs.u[11][i];
    }

    return flops;
}


//
// same, but direct solver
//
extern "C" float external_vel_direct_f_ (const int* nsrc,
                                         const float* sx, const float* sy, const float* sz,
                                         const float* ssx, const float* ssy, const float* ssz,
                                         const float* sr,
                                         const int* ntarg,
                                         const float* tx, const float* ty, const float* tz,
                                         float* tu,  float* tv,  float* tw,
                                         float* tux, float* tvx, float* twx,
                                         float* tuy, float* tvy, float* twy,
                                         float* tuz, float* tvz, float* twz) {
    float flops = 0.0;
    bool silent = true;

    if (!silent) printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double,3,3,12> srcs(*nsrc, true);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[2][i] = sz[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[0][i] = ssx[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[1][i] = ssy[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[2][i] = ssz[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<float,double,3,3,12> targs(*ntarg, false);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
    for (int i=0; i<*ntarg; ++i) targs.x[2][i] = tz[i];
    for (auto& u : targs.u[0]) { u = 0.0f; }
    for (auto& u : targs.u[1]) { u = 0.0f; }
    for (auto& u : targs.u[2]) { u = 0.0f; }
    for (auto& u : targs.u[3]) { u = 0.0f; }
    for (auto& u : targs.u[4]) { u = 0.0f; }
    for (auto& u : targs.u[5]) { u = 0.0f; }
    for (auto& u : targs.u[6]) { u = 0.0f; }
    for (auto& u : targs.u[7]) { u = 0.0f; }
    for (auto& u : targs.u[8]) { u = 0.0f; }
    for (auto& u : targs.u[9]) { u = 0.0f; }
    for (auto& u : targs.u[10]) { u = 0.0f; }
    for (auto& u : targs.u[11]) { u = 0.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    if (!silent) printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());

    //
    // Run a naive O(N^2) summation
    //
    if (!silent) printf("\nRun the direct O(N^2) summation\n");
    start = std::chrono::system_clock::now();
    flops += nbody_naive(srcs, targs, 1);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    double dt = elapsed_seconds.count();
    if (!silent) printf("  direct summations:\t\t[%.4f] seconds\n\n", dt);

    // save the results out
    for (int i=0; i<*ntarg; ++i) tu[i] += targs.u[0][i];
    for (int i=0; i<*ntarg; ++i) tv[i] += targs.u[1][i];
    for (int i=0; i<*ntarg; ++i) tw[i] += targs.u[2][i];
    for (int i=0; i<*ntarg; ++i) tux[i] += targs.u[3][i];
    for (int i=0; i<*ntarg; ++i) tvx[i] += targs.u[4][i];
    for (int i=0; i<*ntarg; ++i) twx[i] += targs.u[5][i];
    for (int i=0; i<*ntarg; ++i) tuy[i] += targs.u[6][i];
    for (int i=0; i<*ntarg; ++i) tvy[i] += targs.u[7][i];
    for (int i=0; i<*ntarg; ++i) twy[i] += targs.u[8][i];
    for (int i=0; i<*ntarg; ++i) tuz[i] += targs.u[9][i];
    for (int i=0; i<*ntarg; ++i) tvz[i] += targs.u[10][i];

    return flops;
}
