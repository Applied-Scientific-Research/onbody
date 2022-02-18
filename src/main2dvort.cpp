/*
 * main2dvort.cpp - driver for Barnes-Hut treecode
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
#include <vector>
#include <iostream>
#include <chrono>

// the fast solver
extern "C" float external_vel_solver_f_ (const int*, const float*, const float*,
                                                     const float*, const float*,
                                         const int*, const float*, const float*,
                                                           float*,       float*);

// the direct solver
extern "C" float external_vel_direct_f_ (const int*, const float*, const float*,
                                                     const float*, const float*,
                                         const int*, const float*, const float*,
                                                           float*,       float*);

const char* progname = "main2dvort";

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

    size_t numSrcs = 10000;
    size_t numTargs = 10000;
    bool compareToDirect = true;

    if (argc > 1) {
        if (strncmp(argv[1], "-n=", 3) == 0) {
            size_t num = atof(argv[1] + 3);
            if (num < 1) usage();
            numSrcs = num;
            numTargs = num;
        }
    }

    printf("Running %s with %ld sources and %ld targets\n", progname, numSrcs, numTargs);
    auto start = std::chrono::system_clock::now();

    // set up the problem

    std::vector<float> sx(numSrcs);
    std::vector<float> sy(numSrcs);
    std::vector<float> ss(numSrcs);
    std::vector<float> sr(numSrcs);
    for (auto& _x : sx) { _x = (float)rand()/(float)RAND_MAX; }
    for (auto& _y : sy) { _y = (float)rand()/(float)RAND_MAX; }
    for (auto& _r : sr) { _r = 1.0f / std::sqrt(numSrcs); }
    // totally random (unrealistic, but worst-case)
    //for (auto& _m : ss) { _m = (-1.0f + 2.0f*(float)rand()/(float)RAND_MAX) / std::sqrt((float)numSrcs); }
    // more realistic
    //const float factor = 1.0 / std::sqrt((float)numSrcs);
    const float factor = 1.0 / (float)numSrcs;
    for (size_t i=0; i<numSrcs; i++) {
        const float dist = std::sqrt(std::pow(sx[i]-0.5,2)+std::pow(sy[i]-0.5,2));
        ss[i] = factor * std::cos(30.0*std::sqrt(dist)) / (5.0*dist+1.0);
        ss[i] *= 0.75 + 0.5*(float)rand()/(float)RAND_MAX;
    }

    std::vector<float> tx(numTargs);
    std::vector<float> ty(numTargs);
    std::vector<float> tu(numTargs);
    std::vector<float> tv(numTargs);
    for (auto& _x : tx) { _x = (float)rand()/(float)RAND_MAX; }
    for (auto& _y : ty) { _y = (float)rand()/(float)RAND_MAX; }
    for (auto& _u : tu) { _u = 0.0f; }
    for (auto& _v : tv) { _v = 0.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    float flops = (float)numSrcs*12.f + (float)numTargs*4.f;
    float gflops = 1.e-9 * flops / (float)elapsed_seconds.count();
    printf("    problem setup:\t\t[%.4f] seconds at %.3f GFlop/s\n", (float)elapsed_seconds.count(), gflops);

    // and call the solver

    start = std::chrono::system_clock::now();
    int ns = numSrcs;
    int nt = numTargs;
    flops = external_vel_solver_f_(&ns, sx.data(), sy.data(), ss.data(), sr.data(),
                                   &nt, tx.data(), ty.data(), tu.data(), tv.data());

    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    gflops = 1.e-9 * flops / (float)elapsed_seconds.count();
    printf("    external_vel_solver_f_:\t[%.4f] seconds at %.3f GFlop/s\n", (float)elapsed_seconds.count(), gflops);

    // compare results to a direct solve

    if (compareToDirect) {

        // run a subset direct solve
        size_t ntskip = std::max(1, (int)((float)numSrcs*(float)numTargs/1.e+10));
        int ntn = numTargs / ntskip;
        std::vector<float> txn(ntn);
        std::vector<float> tyn(ntn);
        std::vector<float> tun(ntn);
        std::vector<float> tvn(ntn);
        for (size_t i=0; i<(size_t)ntn; ++i) {
            const size_t ifast = i * ntskip;
            txn[i] = tx[ifast];
            tyn[i] = ty[ifast];
            tun[i] = 0.0f;
            tvn[i] = 0.0f;
        }
        start = std::chrono::system_clock::now();
        flops = external_vel_direct_f_(&ns, sx.data(), sy.data(), ss.data(), sr.data(),
                                       &ntn, txn.data(), tyn.data(), tun.data(), tvn.data());
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        gflops = 1.e-9 * flops / (float)elapsed_seconds.count();
        printf("    external_vel_direct_f_:\t[%.4f] seconds at %.3f GFlop/s\n", (float)ntskip*(float)elapsed_seconds.count(), gflops);

        // compute the error
        float errsum = 0.0;
        float errcnt = 0.0;
        float maxerr = 0.0;
        for (size_t i=0; i<(size_t)ntn; ++i) {
            const size_t ifast = i * ntskip;
            float thiserr = tu[ifast]-tun[i];
            //if (i<4) printf("      %ld %ld  %g %g   %g %g\n", i, ifast, tx[ifast], txn[i], tu[ifast], tun[i]);
            //printf("      %ld %ld  %g %g   %g %g\n", i, ifast, tx[ifast], txn[i], tu[ifast], tun[i]);
            errsum += thiserr*thiserr;
            //if (thiserr*thiserr > maxerr) printf("      %ld %ld  %g %g   %g %g  new max\n", i, ifast, tx[ifast], txn[i], tu[ifast], tun[i]);
            if (thiserr*thiserr > maxerr) maxerr = thiserr*thiserr;
            errcnt += tun[i]*tun[i];
        }
        printf("    rms error in fast solver:\t%g\n", std::sqrt(errsum/errcnt));
        printf("    max error in fast solver:\t%g\n", std::sqrt(maxerr/(errcnt/(float)ntn)));
    }

    return 0;
}
