/*
 * barneshut.cpp - driver for Barnes-Hut treecode
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 */

#include <barneshut.h>

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


const char* progname = "barneshut";

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

    static std::vector<int> test_iterations = {1, 1, 1};
    bool just_build_trees = false;
    size_t numSrcs = 10000;
    size_t numTargs = 10000;

    if (argc > 1) {
        if (strncmp(argv[1], "-n=", 3) == 0) {
            size_t num = atof(argv[1] + 3);
            if (num < 1) usage();
            numSrcs = num;
            numTargs = num;
        }
    }

    printf("Running %s with %ld sources and %ld targets\n\n", progname, numSrcs, numTargs);

    // if problem is too big, skip some number of target particles
    size_t ntskip = std::max(1, (int)((float)numSrcs*(float)numTargs/2.e+9));

    printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double> srcs(numSrcs);
    // initialize particle data
    srcs.random_in_cube();
    srcs.smooth_strengths();

    Parts<float,double> targs(numTargs);
    // initialize particle data
    targs.random_in_cube();
    for (auto& m : targs.m) { m = 1.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // allocate and initialize tree
    printf("\nBuilding the source tree\n");
    start = std::chrono::system_clock::now();
    Tree<float> stree(numSrcs);
    printf("  with %ld particles and block size of %ld\n", numSrcs, blockSize);
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

    // first, reorder tree until all parts are adjacent in space-filling curve
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


    if (just_build_trees) exit(0);

    //
    // Run the O(N^2) implementation
    //
    printf("\nRun the naive O(N^2) method (every %ld particles)\n", ntskip);
    double minNaive = 1e30;
    for (int i = 0; i < test_iterations[0]; ++i) {
        start = std::chrono::system_clock::now();
        nbody_naive(srcs, targs, ntskip);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count() * (float)ntskip;
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minNaive = std::min(minNaive, dt);
    }
    printf("[onbody naive]:\t\t\t[%.4f] seconds\n", minNaive);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g\n",i,targs.u[i],targs.v[i]);
    std::vector<float> naiveu(targs.u.begin(), targs.u.end());

    float errsum = 0.0;
    float errcnt = 0.0;

    //
    // Run a simple O(NlogN) treecode - boxes approximate as particles
    //
    if (test_iterations[1] > 0) {
    printf("\nRun the treecode O(NlogN)\n");
    double minTreecode = 1e30;
    for (int i = 0; i < test_iterations[1]; ++i) {
        start = std::chrono::system_clock::now();
        nbody_treecode1(srcs, stree, targs, 8.0f);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minTreecode = std::min(minTreecode, dt);
    }
    printf("[onbody treecode]:\t\t[%.4f] seconds\n", minTreecode);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g\n",i,targs.u[i],targs.v[i]);
    // save the results for comparison
    std::vector<float> treecodeu(targs.u.begin(), targs.u.end());

    // compare accuracy
    errsum = 0.0;
    errcnt = 0.0;
    for (size_t i=0; i< targs.u.size(); i+=ntskip) {
        float thiserr = treecodeu[i]-naiveu[i];
        errsum += thiserr*thiserr;
        errcnt += naiveu[i]*naiveu[i];
    }
    printf("RMS error in treecode is %g\n", std::sqrt(errsum/errcnt));
    }


    //
    // Run a better O(NlogN) treecode - boxes use equivalent particles
    //
    if (test_iterations[2] > 0) {
    printf("\nRun the treecode O(NlogN) with equivalent particles\n");
    double minTreecode2 = 1e30;
    for (int i = 0; i < test_iterations[2]; ++i) {
        start = std::chrono::system_clock::now();
        nbody_treecode2(srcs, eqsrcs, stree, targs, 1.3f);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minTreecode2 = std::min(minTreecode2, dt);
    }
    printf("[onbody treecode2]:\t\t[%.4f] seconds\n", minTreecode2);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g\n",i,targs.u[i],targs.v[i]);
    // save the results for comparison
    std::vector<float> treecodeu2(targs.u.begin(), targs.u.end());

    // compare accuracy
    errsum = 0.0;
    errcnt = 0.0;
    for (size_t i=0; i< targs.u.size(); i+=ntskip) {
        float thiserr = treecodeu2[i]-naiveu[i];
        errsum += thiserr*thiserr;
        errcnt += naiveu[i]*naiveu[i];
    }
    printf("RMS error in treecode2 is %g\n", std::sqrt(errsum/errcnt));
    }

    return 0;
}
