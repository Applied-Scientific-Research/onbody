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

extern float external_vel_solver_f (const int*, const float*, const float*,
                                                const float*, const float*,
                                    const int*, const float*, const float*,
                                                      float*,       float*);

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
    auto start = std::chrono::system_clock::now();

    std::vector<float> sx(numSrcs);
    std::vector<float> sy(numSrcs);
    std::vector<float> ss(numSrcs);
    std::vector<float> sr(numSrcs);
    for (auto& _x : sx) { _x = (float)rand()/(float)RAND_MAX; }
    for (auto& _y : sy) { _y = (float)rand()/(float)RAND_MAX; }
    for (auto& _m : ss) { _m = (-1.0f + 2.0f*(float)rand()/(float)RAND_MAX) / std::sqrt((float)numSrcs); }
    for (auto& _r : sr) { _r = 1.0f / std::sqrt(numSrcs); }

    std::vector<float> tx(numTargs);
    std::vector<float> ty(numTargs);
    std::vector<float> tu(numTargs);
    std::vector<float> tv(numTargs);
    for (auto& _x : tx) { _x = (float)rand()/(float)RAND_MAX; }
    for (auto& _y : ty) { _y = (float)rand()/(float)RAND_MAX; }
    for (auto& _u : tu) { _u = 0.0f; }
    for (auto& _v : tv) { _v = 0.0f; }

    int ns = numSrcs;
    int nt = numTargs;
    float flops = external_vel_solver_f(&ns, sx.data(), sy.data(), ss.data(), sr.data(),
                                        &nt, tx.data(), ty.data(), tu.data(), tv.data());

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    const float gflops = 1.e-9 * flops / (float)elapsed_seconds.count();
    printf("    external_vel_solver_f_:\t[%.4f] seconds at %.3f GFlop/s\n", (float)elapsed_seconds.count(), gflops);


    return 0;
}
