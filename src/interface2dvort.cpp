/*
 * interface2dvort.cpp - interface code for Barnes-Hut treecode
 *
 * Copyright (c) 2017-22, Mark J Stock <markjstock@gmail.com>
 */

#define STORE float
#define ACCUM float

#include "CoreFunc2d.hpp"
#include "BarycentricLagrange.hpp"

#ifdef USE_VC
#include <Vc/Vc>
#endif

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

#ifdef USE_VC
template <class S> using Vector = std::vector<S, Vc::Allocator<S>>;
#else
template <class S> using Vector = std::vector<S>;
#endif


//
// The inner, scalar kernel
//
template <class S, class A>
static inline void nbody_kernel(const S sx, const S sy,
                                const S sr, const S ss,
                                const S tx, const S ty,
                                A& __restrict__ tu, A& __restrict__ tv) {
    // 12 flops
    const S dx = tx - sx;
    const S dy = ty - sy;
    const S r2 = ss * core_func<S>(dx*dx + dy*dy, sr);
    tu -= mycast<S,A>(r2*dy);
    tv += mycast<S,A>(r2*dx);
}

static inline int nbody_kernel_flops() { return 10 + flops_tp_nograds(); }

template <class S, class A, int PD, int SD, int OD> class Parts;

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);

#ifdef USE_VC
    // this works
    Vc::simdize<Vector<STORE>::const_iterator> sxit, syit, ssit, srit;
    // but this does not
    //Vc::simdize<Vector<S>::const_iterator> sxit, syit, ssit, srit;
    const size_t nSrcVec = (jend-jstart + Vc::Vector<S>::Size - 1) / Vc::Vector<S>::Size;

    // a simd type for A with the same number of entries as S
    typedef Vc::SimdArray<A,Vc::Vector<S>::size()> VecA;

    // spread this target over a vector
    const Vc::Vector<S> vtx = targs.x[0][i];
    const Vc::Vector<S> vty = targs.x[1][i];
    VecA vtu0(0.0f);
    VecA vtu1(0.0f);
    // reference source data as Vc::Vector<A>
    sxit = srcs.x[0].begin() + jstart;
    syit = srcs.x[1].begin() + jstart;
    ssit = srcs.s[0].begin() + jstart;
    srit = srcs.r.begin() + jstart;
    for (size_t j=0; j<nSrcVec; ++j) {
        nbody_kernel<Vc::Vector<S>,VecA>(
                     *sxit, *syit, *srit, *ssit,
                     vtx, vty, vtu0, vtu1);
        // advance the source iterators
        ++sxit;
        ++syit;
        ++ssit;
        ++srit;
    }
    // reduce target results to scalar
    targs.u[0][i] += vtu0.sum();
    targs.u[1][i] += vtu1.sum();

#else
    for (size_t j=jstart; j<jend; ++j) {
        nbody_kernel<S,A>(srcs.x[0][j], srcs.x[1][j], srcs.r[j], srcs.s[0][j],
                     targs.x[0][i], targs.x[1][i],
                     targs.u[0][i], targs.u[1][i]);
    }
#endif
}

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t istart, const size_t iend) {
    //printf("    compute srcs %ld-%ld on targs %ld-%ld\n", jstart, jend, istart, iend);

#ifdef USE_VC
    Vc::simdize<Vector<STORE>::const_iterator> sxit, syit, ssit, srit;
    const size_t nSrcVec = (jend-jstart + Vc::Vector<S>::Size - 1) / Vc::Vector<S>::Size;

    // a simd type for A with the same number of entries as S
    typedef Vc::SimdArray<A,Vc::Vector<S>::size()> VecA;

    for (size_t i=istart; i<iend; ++i) {
        // spread this target over a vector
        const Vc::Vector<S> vtx = targs.x[0][i];
        const Vc::Vector<S> vty = targs.x[1][i];
        VecA vtu0(0.0f);
        VecA vtu1(0.0f);
        // convert source data to Vc::Vector<S>
        sxit = srcs.x[0].begin() + jstart;
        syit = srcs.x[1].begin() + jstart;
        ssit = srcs.s[0].begin() + jstart;
        srit = srcs.r.begin() + jstart;
        for (size_t j=0; j<nSrcVec; ++j) {
            nbody_kernel<Vc::Vector<S>,VecA>(
                         *sxit, *syit, *srit, *ssit,
                         vtx, vty, vtu0, vtu1);
            if (false and (std::abs(vtu0.sum())>100.0 or std::abs(vtu1.sum())>100.0)) {
                std::cout << "die in ppinter targ " << i << " srcvec " << j << "\n " << *sxit << "\n " << *syit << "\n " << *ssit << "\n " << *srit << "\n";
                for (size_t k=0; k<Vc::Vector<S>::Size; ++k) {
                    size_t idx = Vc::Vector<S>::Size * (sxit - srcs.x[0].begin()) + k;
                    std::cout << "   src " << idx << " " << srcs.s[0][idx] << " " << srcs.r[idx] << "\n";
                }
                exit(0);
            }
            // advance the source iterators
            ++sxit;
            ++syit;
            ++ssit;
            ++srit;
        }
        // reduce target results to scalar
        targs.u[0][i] += vtu0.sum();
        targs.u[1][i] += vtu1.sum();
    }
#else
    for (size_t i=istart; i<iend; ++i) {
        for (size_t j=jstart; j<jend; ++j) {
            nbody_kernel<S,A>(srcs.x[0][j], srcs.x[1][j], srcs.r[j], srcs.s[0][j],
                         targs.x[0][i], targs.x[1][i],
                         targs.u[0][i], targs.u[1][i]);
        }
    }
#endif
}

template <class S, int PD, int SD> class Tree;

template <class S, class A, int PD, int SD, int OD>
void tpinter(const Tree<S,PD,SD>& __restrict__ stree, const size_t j,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    nbody_kernel<S,A>(stree.x[0][j], stree.x[1][j], stree.pr[j], stree.s[0][j],
                 targs.x[0][i], targs.x[1][i],
                 targs.u[0][i], targs.u[1][i]);
}

//
// Now we can include the tree-building and recursion code
//
#include "barneshut.hpp"
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
    // 1.2 and 3 for 1e-3 error (float-float OK)
    // 1.3 and 4 for 1e-4 error (float-float OK)
    // 1.4 and 5 for 1e-5 error (double-double, though)
    // 1.5 and 6 for 1e-6 error (double-double, though)
    const float theta = 1.3;
    const int32_t order = 4;
    const bool silent = true;
    const bool blockwise = true;

    if (!silent) printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<STORE,ACCUM,2,1,2> srcs(*nsrc, true);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[0][i] = ss[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    Parts<STORE,ACCUM,2,1,2> targs(*ntarg, false);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
    for (auto& u : targs.u[0]) { u = 0.0f; }
    for (auto& u : targs.u[1]) { u = 0.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    if (!silent) printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // initialize and generate tree
    if (!silent) printf("\nBuilding the source tree\n");
    if (!silent) printf("  with %d particles and block size of %ld\n", *nsrc, blockSize);
    start = std::chrono::system_clock::now();
    Tree<STORE,2,1> stree(0);
    // split this node and recurse
    (void) makeTree(srcs, stree);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());

    if (order < 0) {
        // first, reorder tree until all parts are adjacent in space-filling curve
        start = std::chrono::system_clock::now();
        #pragma omp parallel
        #pragma omp single
        (void) refineTree(srcs, stree, 1);
        #pragma omp taskwait
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        if (!silent) printf("  refine within leaf nodes:\t[%.4f] seconds\n", elapsed_seconds.count());
    }

    // buffer source arrays to accommodate vector length
    (void) srcs.buffer_end(VecSize<STORE>);

    // find equivalent particles
    if (!silent) printf("\nCalculating equivalent particles\n");
    start = std::chrono::system_clock::now();
    Parts<STORE,ACCUM,2,1,2> eqsrcs((stree.numnodes/2) * blockSize, true);
    if (!silent) printf("  need %ld particles\n", eqsrcs.n);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    if (!silent) printf("  allocate eqsrcs structures:\t[%.4f] seconds\n", elapsed_seconds.count());

    // generate the far-field approximations
    if (order < 0) {
        // then, march through arrays merging pairs as you go up
        start = std::chrono::system_clock::now();
        (void) calcEquivalents(srcs, eqsrcs, stree, 1);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        if (!silent) printf("  create equivalent parts:\t[%.4f] seconds\n", elapsed_seconds.count());
    } else {
        // upward pass to compute barycentric lagrange particles
        start = std::chrono::system_clock::now();
        (void) calcBarycentricLagrange(srcs, eqsrcs, stree, order, 1);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        if (!silent) printf("  create barylagrange parts:\t[%.4f] seconds\n", elapsed_seconds.count());
    }


    // don't need the target tree for treecode, but will for boxwise and fast code
    Tree<STORE,2,1> ttree(0);
    if (blockwise) {
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
        flops += nbody_treecode3(srcs, eqsrcs, stree, targs, ttree, theta);
    } else {
        flops += nbody_treecode2(srcs, eqsrcs, stree, targs, theta);
    }

    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    double dt = elapsed_seconds.count();
    if (!silent) printf("  treecode summations:\t\t[%.4f] seconds\n\n", dt);


    // pull results from the object
    if (blockwise) {
        // need to rearrange the results back in original order
        for (int i=0; i<*ntarg; ++i) tu[targs.gidx[i]] += targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[targs.gidx[i]] += targs.u[1][i];
        if (!silent) for (int i=0; i<std::min(10,*ntarg); ++i) printf("  %d  %g %g  %g %g\n", i, tx[i], ty[i], tu[i], tv[i]);
        //if (!silent) for (int i=0; i<*ntarg; ++i) printf("  %d %ld  %g %g\n", i, targs.gidx[i], tu[i], tv[i]);
        //for (int i=0; i<*nsrc; i+=100) printf("  %d  %g %g  %g %g\n", i, sx[i], sy[i], ss[i], sr[i]);
        //for (int i=0; i<*ntarg; i+=100) printf("  %d %ld  %g %g\n", i, targs.gidx[i], tu[i], tv[i]);
    } else {
        // pull them out directly
        for (int i=0; i<*ntarg; ++i) tu[i] += targs.u[0][i];
        for (int i=0; i<*ntarg; ++i) tv[i] += targs.u[1][i];
        //if (!silent) for (int i=0; i<*ntarg; i+=10) printf("  %d  %g %g\n", i, tu[i], tv[i]);
        if (!silent) for (int i=0; i<std::min(10,*ntarg); ++i) printf("  %d  %g %g  %g %g\n", i, tx[i], ty[i], tu[i], tv[i]);
    }

    return flops;
}


//
// same, but direct solver
//
extern "C" float external_vel_direct_f_ (const int* nsrc,
                                         const float* sx, const float* sy,
                                         const float* ss, const float* sr,
                                         const int* ntarg,
                                         const float* tx, const float* ty,
                                               float* tu,       float* tv) {
    float flops = 0.0;
    const bool silent = true;

    if (!silent) printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<STORE,ACCUM,2,1,2> srcs(*nsrc, true);
    // initialize particle data
    for (int i=0; i<*nsrc; ++i) srcs.x[0][i] = sx[i];
    for (int i=0; i<*nsrc; ++i) srcs.x[1][i] = sy[i];
    for (int i=0; i<*nsrc; ++i) srcs.s[0][i] = ss[i];
    for (int i=0; i<*nsrc; ++i) srcs.r[i] = sr[i];

    // buffer source arrays to accommodate vector length
    (void) srcs.buffer_end(VecSize<STORE>);

    Parts<STORE,ACCUM,2,1,2> targs(*ntarg, false);
    // initialize particle data
    for (int i=0; i<*ntarg; ++i) targs.x[0][i] = tx[i];
    for (int i=0; i<*ntarg; ++i) targs.x[1][i] = ty[i];
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

    if (!silent) for (int i=0; i<std::min(10,*ntarg); ++i) printf("  %d  %g %g  %g %g\n", i, tx[i], ty[i], tu[i], tv[i]);

    return flops;
}
