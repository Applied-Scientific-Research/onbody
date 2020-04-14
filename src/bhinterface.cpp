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
float external_vel_solver_f (const int* nsrc,  const float* sx, const float* sy,
                                               const float* ss, const float* sr,
                             const int* ntarg, const float* tx, const float* ty,
                                                     float* tu,       float* tv) {

    float flops = -1.0;

    printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double> srcs(*nsrc);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.y[i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.m[i] = ss[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<float,double> targs(*ntarg);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.y[i] = ty[i];
    for (auto& m : targs.m) { m = 1.0f; }
    for (int i=0; i<*ntarg; ++i) targs.u[i] = tu[i];
    for (int i=0; i<*ntarg; ++i) targs.v[i] = tv[i];
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // allocate and initialize tree
    printf("\nBuilding the source tree\n");
    start = std::chrono::system_clock::now();
    Tree<float> stree(*nsrc);
    printf("  with %d particles and block size of %ld\n", *nsrc, blockSize);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  allocate and init tree:\t[%.4f] seconds\n", elapsed_seconds.count());

    // split this node and recurse
    start = std::chrono::system_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) splitNode(srcs, 0, srcs.n, stree, 1);
    #pragma omp taskwait
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());

    // find equivalent particles
    printf("\nCalculating equivalent particles\n");
    start = std::chrono::system_clock::now();
    Parts<float,double> eqsrcs((stree.numnodes/2) * blockSize);
    printf("  need %ld particles\n", eqsrcs.n);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  allocate eqsrcs structures:\t[%.4f] seconds\n", elapsed_seconds.count());

    // reorder tree until all parts are adjacent in space-filling curve
    start = std::chrono::system_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) refineTree(srcs, stree, 1);
    #pragma omp taskwait
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  refine within leaf nodes:\t[%.4f] seconds\n", elapsed_seconds.count());
    //for (size_t i=0; i<stree.num[1]; ++i)
    //    printf("%d %g %g %g\n", i, srcs.x[i], srcs.y[i], srcs.z[i]);

    // then, march through arrays merging pairs as you go up
    start = std::chrono::system_clock::now();
    (void) calcEquivalents(srcs, eqsrcs, stree, 1);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  create equivalent parts:\t[%.4f] seconds\n", elapsed_seconds.count());


    //
    // Run a better O(NlogN) treecode - boxes use equivalent particles
    //
    printf("\nRun the treecode O(NlogN) with equivalent particles\n");
    start = std::chrono::system_clock::now();
    nbody_treecode2(srcs, eqsrcs, stree, targs, 1.3f);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    double dt = elapsed_seconds.count();
    printf("  treecode summations:\t\t[%.4f] seconds\n\n", dt);

    // save the results for comparison
    for (int i=0; i<*ntarg; ++i) {
      tu[i] = targs.u[i];
      tv[i] = targs.v[i];
    }

    return flops;
}
