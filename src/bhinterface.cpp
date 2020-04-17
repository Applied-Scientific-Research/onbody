/*
 * bhinterface.cpp - interface code for Barnes-Hut treecode
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 */

#include <barneshut.h>

#include <cstdlib>
#include <cstdint>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif


//
// call this function from an external program
//
extern "C" float external_vel_solver_f_ (const int* nsrc,  const float* sx, const float* sy,
                                               const float* ss, const float* sr,
                             const int* ntarg, const float* tx, const float* ty,
                                                     float* tu,       float* tv) {
    float flops = 0.0;
    const bool silent = false;
    const bool createTargTree = true;
    const bool blockwise = true;

    if (!silent) printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double,2> srcs(*nsrc);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.m[i] = ss[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<float,double,2> targs(*ntarg);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
    for (auto& m : targs.m) { m = 1.0f/(float)(*ntarg); }
    for (int i=0; i<*ntarg; ++i) targs.u[0][i] = tu[i];
    for (int i=0; i<*ntarg; ++i) targs.u[1][i] = tv[i];
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    if (!silent) printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // allocate and initialize tree
    if (!silent) printf("\nBuilding the source tree\n");
    if (!silent) printf("  with %d particles and block size of %ld\n", *nsrc, blockSize);
    start = std::chrono::system_clock::now();
    Tree<float,2> stree(*nsrc);
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
    Parts<float,double,2> eqsrcs((stree.numnodes/2) * blockSize);
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


    Tree<float,2> ttree(0);
    if (createTargTree or blockwise) {
        if (!silent) printf("\nBuilding the target tree\n");
        if (!silent) printf("  with %d particles and block size of %ld\n", *ntarg, blockSize);

        start = std::chrono::system_clock::now();
        ttree = Tree<float,2>(*ntarg);
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
        flops += nbody_treecode3(srcs, eqsrcs, stree, targs, ttree, 4.0f);
    } else {
        flops += nbody_treecode2(srcs, eqsrcs, stree, targs, 4.0f);
    }

    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    double dt = elapsed_seconds.count();
    if (!silent) printf("  treecode summations:\t\t[%.4f] seconds\n\n", dt);


    // pull results from the object
    if (createTargTree) {
        // need to rearrange the results back in original order
        for (int i=0; i<*ntarg; ++i) tu[targs.gidx[i]] = targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[targs.gidx[i]] = targs.u[1][i];
    } else {
        // pull them out directly
        for (int i=0; i<*ntarg; ++i) tu[i] = targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[i] = targs.u[1][i];
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
    Parts<float,double,2> srcs(*nsrc);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.m[i] = ss[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<float,double,2> targs(*ntarg);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
    for (auto& m : targs.m) { m = 1.0f/(float)(*ntarg); }
    for (int i=0; i<*ntarg; ++i) targs.u[0][i] = tu[i];
    for (int i=0; i<*ntarg; ++i) targs.u[1][i] = tv[i];
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
    for (int i=0; i<*ntarg; ++i) tu[i] = targs.u[0][i];
    for (int i=0; i<*ntarg; ++i) tv[i] = targs.u[1][i];

    return flops;
}
