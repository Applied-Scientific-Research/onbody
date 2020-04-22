/*
 * bhinterface.cpp - interface code for Barnes-Hut treecode
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


//
// The inner, scalar kernel
//
template <class S, class A>
static inline void nbody_kernel(const S sx, const S sy,
                                const S sr, const S sm,
                                const S tx, const S ty,
                                A& __restrict__ tax, A& __restrict__ tay) {
    // 12 flops
    const S dx = tx - sx;
    const S dy = ty - sy;
    // Rosenhead-Moore
    S r2 = dx*dx + dy*dy + sr*sr;
    r2 = sm/r2;
    // Exponential
    //const S d2 = dx*dx + dy*dy + (S)1.e-14;
    //S r2 = sm * ((S)1.0 - std::exp(-d2/(sr*sr))) / d2;
    tax -= r2 * dy;
    tay += r2 * dx;
}

static inline int nbody_kernel_flops() { return 12; }

template <class S, class A, int PD, int SD, int OD> class Parts;

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    for (size_t j=jstart; j<jend; ++j) {
        nbody_kernel(srcs.x[0][j],  srcs.x[1][j], srcs.r[j], srcs.s[0][j],
                     targs.x[0][i], targs.x[1][i],
                     targs.u[0][i], targs.u[1][i]);
    }
}

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t istart, const size_t iend) {
    //printf("    compute srcs %ld-%ld on targs %ld-%ld\n", jstart, jend, istart, iend);
    for (size_t i=istart; i<iend; ++i) {
        for (size_t j=jstart; j<jend; ++j) {
            nbody_kernel(srcs.x[0][j],  srcs.x[1][j], srcs.r[j], srcs.s[0][j],
                         targs.x[0][i], targs.x[1][i],
                         targs.u[0][i], targs.u[1][i]);
    }
    }
}

template <class S, int PD, int SD> class Tree;

template <class S, class A, int PD, int SD, int OD>
void tpinter(const Tree<S,PD,SD>& __restrict__ stree, const size_t j,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    nbody_kernel(stree.x[0][j], stree.x[1][j], stree.pr[j], stree.s[0][j],
                 targs.x[0][i], targs.x[1][i],
                 targs.u[0][i], targs.u[1][i]);
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
                                         const float* sx, const float* sy,
                                         const float* ss, const float* sr,
                                         const int* ntarg,
                                         const float* tx, const float* ty,
                                               float* tu,       float* tv) {
    float flops = 0.0;
    const bool silent = false;
    const bool createTargTree = true;
    const bool blockwise = true;

    if (!silent) printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double,2,1,2> srcs(*nsrc);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[0][i] = ss[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<float,double,2,1,2> targs(*ntarg);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
    for (auto& m : targs.s[0]) { m = 1.0f/(float)(*ntarg); }
    for (auto& u : targs.u[0]) { u = 0.0f; }
    for (auto& u : targs.u[1]) { u = 0.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    if (!silent) printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // allocate and initialize tree
    if (!silent) printf("\nBuilding the source tree\n");
    if (!silent) printf("  with %d particles and block size of %ld\n", *nsrc, blockSize);
    start = std::chrono::system_clock::now();
    Tree<float,2,1> stree(*nsrc);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  allocate and init tree:\t[%.4f] seconds\n", elapsed_seconds.count());

    // split this node and recurse
    start = std::chrono::system_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) splitNode(srcs, 0, srcs.n, stree, 1);
    #pragma omp taskwait
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());

    // find equivalent particles
    if (!silent) printf("\nCalculating equivalent particles\n");
    start = std::chrono::system_clock::now();
    Parts<float,double,2,1,2> eqsrcs((stree.numnodes/2) * blockSize);
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


    Tree<float,2,1> ttree(0);
    if (createTargTree or blockwise) {
        if (!silent) printf("\nBuilding the target tree\n");
        if (!silent) printf("  with %d particles and block size of %ld\n", *ntarg, blockSize);

        start = std::chrono::system_clock::now();
        ttree = Tree<float,2,1>(*ntarg);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        if (!silent) printf("  allocate and init tree:\t[%.4f] seconds\n", elapsed_seconds.count());

        // split this node and recurse
        start = std::chrono::system_clock::now();
        #pragma omp parallel
        #pragma omp single
        (void) splitNode(targs, 0, targs.n, ttree, 1);
        #pragma omp taskwait
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        if (!silent) printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());
    }


    //
    // Run a better O(NlogN) treecode - boxes use equivalent particles
    //
    if (!silent) printf("\nRun the treecode O(NlogN) with equivalent particles\n");
    start = std::chrono::system_clock::now();

    if (blockwise) {
        flops += nbody_treecode3(srcs, eqsrcs, stree, targs, ttree, 1.6f);
    } else {
        flops += nbody_treecode2(srcs, eqsrcs, stree, targs, 1.5f);
    }

    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    double dt = elapsed_seconds.count();
    if (!silent) printf("  treecode summations:\t\t[%.4f] seconds\n\n", dt);


    // pull results from the object
    if (createTargTree) {
        // need to rearrange the results back in original order
        for (int i=0; i<*ntarg; ++i) tu[targs.gidx[i]] += targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[targs.gidx[i]] += targs.u[1][i];
        //for (int i=0; i<*ntarg; ++i) printf("  %d %ld  %g %g\n", i, targs.gidx[i], tu[i], tv[i]);
    } else {
        // pull them out directly
        for (int i=0; i<*ntarg; ++i) tu[i] += targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[i] += targs.u[1][i];
    }

    return flops;
}


//
// same, but direct solver
//
extern "C" float external_vel_direct_f_ (const int* nsrc,  const float* sx, const float* sy,
                                               const float* ss, const float* sr,
                             const int* ntarg, const float* tx, const float* ty,
                                                     float* tu,       float* tv) {
    float flops = 0.0;
    bool silent = true;

    if (!silent) printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double,2,1,2> srcs(*nsrc);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[0][i] = ss[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<float,double,2,1,2> targs(*ntarg);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
    for (auto& m : targs.s[0]) { m = 1.0f/(float)(*ntarg); }
    for (auto& u : targs.u[0]) { u = 0.0f; }
    for (auto& u : targs.u[1]) { u = 0.0f; }
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

    return flops;
}
