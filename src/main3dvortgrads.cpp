/*
 * main3dvortgrads.cpp - driver for Barnes-Hut treecode
 *
 * Copyright (c) 2017-22, Mark J Stock <markjstock@gmail.com>
 */

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <iostream>
#include <chrono>

// the fast solver
extern "C" float external_vel_solver_f_ (const int*,
                                         const float*, const float*, const float*,
                                         const float*, const float*, const float*,
                                         const float*,
                                         const int*,
                                         const float*, const float*, const float*,
                                         float*,  float*,  float*,
                                         float*, float*, float*,
                                         float*, float*, float*,
                                         float*, float*, float*);

// the direct solver
extern "C" float external_vel_direct_f_ (const int*,
                                         const float*, const float*, const float*,
                                         const float*, const float*, const float*,
                                         const float*,
                                         const int*,
                                         const float*, const float*, const float*,
                                         float*,  float*,  float*,
                                         float*, float*, float*,
                                         float*, float*, float*,
                                         float*, float*, float*);

const char* progname = "main3dvortgrads";

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
    auto start = std::chrono::steady_clock::now();

    // set up the problem

    std::vector<float> sx(numSrcs);
    std::vector<float> sy(numSrcs);
    std::vector<float> sz(numSrcs);
    std::vector<float> ssx(numSrcs);
    std::vector<float> ssy(numSrcs);
    std::vector<float> ssz(numSrcs);
    std::vector<float> sr(numSrcs);
    for (auto& _x : sx) { _x = (float)rand()/(float)RAND_MAX; }
    for (auto& _y : sy) { _y = (float)rand()/(float)RAND_MAX; }
    for (auto& _z : sz) { _z = (float)rand()/(float)RAND_MAX; }
    for (auto& _r : sr) { _r = 1.0f / std::sqrt(numSrcs); }
    // totally random (unrealistic, but worst-case)
    //for (auto& _m : ss) { _m = (-1.0f + 2.0f*(float)rand()/(float)RAND_MAX) / std::sqrt((float)numSrcs); }
    // more realistic
    //const float factor = 1.0 / std::sqrt((float)numSrcs);
    const float factor = 1.0 / (float)numSrcs;
    //for (size_t i=0; i<numSrcs; i++) {
    //    const float dist = std::sqrt(std::pow(sx[i]-0.5,2)+std::pow(sy[i]-0.5,2));
    //    ssx[i] = factor * std::cos(30.0*std::sqrt(dist)) / (5.0*dist+1.0);
     //   ssx[i] *= 0.75 + 0.5*(float)rand()/(float)RAND_MAX;
     //   ssy[i] = factor * std::cos(10.0*std::sqrt(dist)) / (6.0*dist+1.0);
     //   ssy[i] *= 0.75 + 0.5*(float)rand()/(float)RAND_MAX;
     //   ssz[i] = factor * std::cos(50.0*std::sqrt(dist)) / (4.0*dist+1.0);
     //   ssz[i] *= 0.75 + 0.5*(float)rand()/(float)RAND_MAX;
    //}
    for (size_t i=0; i<numSrcs; i++) {
        ssx[i] = factor * std::cos((0+0.7)*10.0*sx[i]);
        ssy[i] = factor * std::cos((1+0.7)*10.0*sy[i]);
        ssz[i] = factor * std::cos((2+0.7)*10.0*sz[i]);
    }

    std::vector<float> tx(numTargs);
    std::vector<float> ty(numTargs);
    std::vector<float> tz(numTargs);
    std::vector<float> tu(numTargs, 0.0f);
    std::vector<float> tv(numTargs, 0.0f);
    std::vector<float> tw(numTargs, 0.0f);
    std::vector<float> tux(numTargs, 0.0f);
    std::vector<float> tvx(numTargs, 0.0f);
    std::vector<float> twx(numTargs, 0.0f);
    std::vector<float> tuy(numTargs, 0.0f);
    std::vector<float> tvy(numTargs, 0.0f);
    std::vector<float> twy(numTargs, 0.0f);
    std::vector<float> tuz(numTargs, 0.0f);
    std::vector<float> tvz(numTargs, 0.0f);
    std::vector<float> twz(numTargs, 0.0f);
    for (auto& _x : tx) { _x = (float)rand()/(float)RAND_MAX; }
    for (auto& _y : ty) { _y = (float)rand()/(float)RAND_MAX; }
    for (auto& _z : tz) { _z = (float)rand()/(float)RAND_MAX; }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    float flops = (float)numSrcs*12.f + (float)numTargs*4.f;
    float gflops = 1.e-9 * flops / (float)elapsed_seconds.count();
    printf("    problem setup:\t\t[%.4f] seconds at %.3f GFlop/s\n", (float)elapsed_seconds.count(), gflops);

    // and call the solver

    start = std::chrono::steady_clock::now();
    int ns = numSrcs;
    int nt = numTargs;
    flops = external_vel_solver_f_(&ns, sx.data(), sy.data(), sz.data(),
                                        ssx.data(), ssy.data(), ssz.data(), sr.data(),
                                   &nt, tx.data(), ty.data(), tz.data(),
                                        tu.data(), tv.data(), tw.data(),
                                        tux.data(), tvx.data(), twx.data(),
                                        tuy.data(), tvy.data(), twy.data(),
                                        tuz.data(), tvz.data(), twz.data());

    end = std::chrono::steady_clock::now();
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
        std::vector<float> tzn(ntn);
        std::vector<float> tun(ntn, 0.0f);
        std::vector<float> tvn(ntn, 0.0f);
        std::vector<float> twn(ntn, 0.0f);
        std::vector<float> tuxn(ntn, 0.0f);
        std::vector<float> tvxn(ntn, 0.0f);
        std::vector<float> twxn(ntn, 0.0f);
        std::vector<float> tuyn(ntn, 0.0f);
        std::vector<float> tvyn(ntn, 0.0f);
        std::vector<float> twyn(ntn, 0.0f);
        std::vector<float> tuzn(ntn, 0.0f);
        std::vector<float> tvzn(ntn, 0.0f);
        std::vector<float> twzn(ntn, 0.0f);
        for (size_t i=0; i<(size_t)ntn; ++i) {
            const size_t ifast = i * ntskip;
            txn[i] = tx[ifast];
            tyn[i] = ty[ifast];
            tzn[i] = tz[ifast];
        }
        start = std::chrono::steady_clock::now();
        flops = external_vel_direct_f_(&ns, sx.data(), sy.data(), sz.data(),
                                            ssx.data(), ssy.data(), ssz.data(), sr.data(),
                                       &ntn, txn.data(), tyn.data(), tzn.data(),
                                             tun.data(), tvn.data(), twn.data(),
                                             tuxn.data(), tvxn.data(), twxn.data(),
                                             tuyn.data(), tvyn.data(), twyn.data(),
                                             tuzn.data(), tvzn.data(), twzn.data());
        end = std::chrono::steady_clock::now();
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
