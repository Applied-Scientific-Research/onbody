/*
 * onvortgrad3d - testbed for an O(N) 3d vortex solver with velocity gradients
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 */

#define STORE float
#define ACCUM float

//#define USE_RM_KERNEL
#define USE_EXPONENTIAL_KERNEL

#ifdef USE_VC
#include <Vc/Vc>
#endif

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

const char* progname = "onvortgrad3d";

#ifdef USE_VC
template <class S> using Vector = std::vector<S, Vc::Allocator<S>>;
#else
template <class S> using Vector = std::vector<S>;
#endif


#ifdef USE_VC
template <class S>
static inline S oor1p5(const S _in) {
  //return Vc::reciprocal(_in*Vc::sqrt(_in));           // 243 GFlop/s
  return Vc::rsqrt(_in) * Vc::reciprocal(_in);          // 302 GFlop/s
}
template <>
inline float oor1p5(const float _in) {
  return 1.0f / (_in*std::sqrt(_in));
}
template <>
inline double oor1p5(const double _in) {
  return 1.0 / (_in*std::sqrt(_in));
}
#else
template <class S>
static inline S oor1p5(const S _in) {
  return S(1.0) / (_in*std::sqrt(_in));
}
#endif

#ifdef USE_VC
template <class S>
static inline S my_recip(const S _in) {
  return Vc::reciprocal(_in);
}
template <>
inline float my_recip(const float _in) {
  return 1.0f / _in;
}
template <>
inline double my_recip(const double _in) {
  return 1.0 / _in;
}
#else
template <class S>
static inline S my_recip(const S _in) {
  return S(1.0) / _in;
}
#endif


#ifdef USE_RM_KERNEL
template <class S>
static inline void core_func (const S distsq, const S sr,
                              S* const __restrict__ r3, S* const __restrict__ bbb) {
  const S r2 = distsq + sr*sr;
  *r3 = oor1p5(r2);
  *bbb = S(-3.0) * (*r3) * my_recip(r2);
}
int flops_tp_grads () { return 7; }
#endif

#ifdef USE_EXPONENTIAL_KERNEL
#ifdef USE_VC
template <class S>
static inline void core_func (const S distsq, const S sr,
                              S* const __restrict__ r3, S* const __restrict__ bbb) {
  const S dm1 = Vc::rsqrt(distsq);
  const S corefac = Vc::reciprocal(sr*sr*sr);
  const S d3 = distsq * distsq * dm1;
  const S reld3 = d3 * corefac;
  // 6 flops to here
  const S dm3 = Vc::reciprocal(d3);

  S myr3, mybbb;
  myr3(reld3 > S(16.0)) = dm3;
  mybbb(reld3 > S(16.0)) = S(-3.0) * dm3 / distsq;
  const S expreld3 = Vc::exp(-reld3);
  myr3(reld3 < S(16.0)) = (S(1.0) - expreld3) * dm3;
  mybbb(reld3 < S(16.0)) = S(3.0) * (corefac*expreld3 - myr3) / distsq;
  myr3(reld3 < S(0.001)) = corefac;
  mybbb(reld3 < S(0.001)) = S(-1.5) * distsq * dm1 * corefac * corefac;
  *r3 = myr3;
  *bbb = mybbb;
}

template <>
inline void core_func (const float distsq, const float sr,
                       float* const __restrict__ r3, float* const __restrict__ bbb) {
  const float dist = std::sqrt(distsq);
  const float corefac = 1.0f / std::pow(sr,3);
  const float d3 = distsq * dist;
  const float reld3 = d3 * corefac;
  // 6 flops to here

  if (reld3 > 16.0f) {
    *r3 = 1.0f / d3;
    *bbb = -3.0f / (d3 * distsq);
    // this is 4 flops and is most likely
  } else if (reld3 < 0.001f) {
    *r3 = corefac;
    *bbb = -1.5f * dist * corefac * corefac;
    // this is 5 flops
  } else {
    const float expreld3 = std::exp(-reld3);
    *r3 = (1.0f - expreld3) / d3;
    *bbb = 3.0f * (corefac*expreld3 - *r3) / distsq;
    // this is 9 flops
  }
}

template <>
inline void core_func (const double distsq, const double sr,
                       double* const __restrict__ r3, double* const __restrict__ bbb) {
  const double dist = std::sqrt(distsq);
  const double corefac = 1.0 / std::pow(sr,3);
  const double d3 = distsq * dist;
  const double reld3 = d3 * corefac;
  // 6 flops to here

  if (reld3 > 16.0) {
    *r3 = 1.0 / d3;
    *bbb = -3.0 / (d3 * distsq);
    // this is 4 flops and is most likely
  } else if (reld3 < 0.001) {
    *r3 = corefac;
    *bbb = -1.5 * dist * corefac * corefac;
    // this is 5 flops
  } else {
    const double expreld3 = std::exp(-reld3);
    *r3 = (1.0 - expreld3) / d3;
    *bbb = 3.0 * (corefac*expreld3 - *r3) / distsq;
    // this is 9 flops
  }
}
#else

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
#endif

int flops_tp_grads () { return 11; }
#endif


// casting from S to A
template <class S, class A>
#ifdef USE_VC
static inline A mycast (const S _in) { return Vc::simd_cast<A>(_in); }
#else
static inline A mycast (const S _in) { return _in; }
#endif
// specialize, in case the non-vectorized version of nbody_kernel is called
template <> inline float mycast (const float _in) { return _in; }
template <> inline double mycast (const float _in) { return (double)_in; }
template <> inline double mycast (const double _in) { return _in; }


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
    S r3, bbb;
    (void) core_func<S>(dx*dx + dy*dy + dz*dz, sr, &r3, &bbb);
    S dxxw = dz*ssy - dy*ssz;
    S dyxw = dx*ssz - dz*ssx;
    S dzxw = dy*ssx - dx*ssy;
    tu += mycast<S,A>(r3*dxxw);
    tv += mycast<S,A>(r3*dyxw);
    tw += mycast<S,A>(r3*dzxw);
    dxxw *= bbb;
    dyxw *= bbb;
    dzxw *= bbb;
    tux += mycast<S,A>(dx*dxxw);
    tvx += mycast<S,A>(dx*dyxw + ssz*r3);
    twx += mycast<S,A>(dx*dzxw - ssy*r3);
    tuy += mycast<S,A>(dy*dxxw - ssz*r3);
    tvy += mycast<S,A>(dy*dyxw);
    twy += mycast<S,A>(dy*dzxw + ssx*r3);
    tuz += mycast<S,A>(dz*dxxw + ssy*r3);
    tvz += mycast<S,A>(dz*dyxw - ssx*r3);
    twz += mycast<S,A>(dz*dzxw);
}

static inline int nbody_kernel_flops() { return 56 + flops_tp_grads(); }

template <class S, class A, int PD, int SD, int OD> class Parts;

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);

#ifdef USE_VC
    // this works
    Vc::simdize<Vector<STORE>::const_iterator> sxit, syit, szit, ssxit, ssyit, sszit, srit;
    // but this does not
    //Vc::simdize<Vector<S>::const_iterator> sxit, syit, szit, ssxit, ssyit, sszit, srit;
    const size_t nSrcVec = (jend-jstart + Vc::Vector<S>::Size - 1) / Vc::Vector<S>::Size;

    // a simd type for A with the same number of entries as S
    typedef Vc::SimdArray<A,Vc::Vector<S>::size()> VecA;

    // spread this target over a vector
    const Vc::Vector<S> vtx = targs.x[0][i];
    const Vc::Vector<S> vty = targs.x[1][i];
    const Vc::Vector<S> vtz = targs.x[2][i];
    VecA vtu0(0.0f);
    VecA vtu1(0.0f);
    VecA vtu2(0.0f);
    VecA vtu3(0.0f);
    VecA vtu4(0.0f);
    VecA vtu5(0.0f);
    VecA vtu6(0.0f);
    VecA vtu7(0.0f);
    VecA vtu8(0.0f);
    VecA vtu9(0.0f);
    VecA vtu10(0.0f);
    VecA vtu11(0.0f);
    // reference source data as Vc::Vector<A>
    sxit = srcs.x[0].begin() + jstart;
    syit = srcs.x[1].begin() + jstart;
    szit = srcs.x[2].begin() + jstart;
    ssxit = srcs.s[0].begin() + jstart;
    ssyit = srcs.s[1].begin() + jstart;
    sszit = srcs.s[2].begin() + jstart;
    srit = srcs.r.begin() + jstart;
    for (size_t j=0; j<nSrcVec; ++j) {
        nbody_kernel<Vc::Vector<S>,VecA>(
                     *sxit, *syit, *szit, *ssxit, *ssyit, *sszit, *srit,
                     vtx, vty, vtz, vtu0, vtu1, vtu2, vtu3, vtu4,
                     vtu5, vtu6, vtu7, vtu8, vtu9, vtu10, vtu11);
        // advance the source iterators
        ++sxit;
        ++syit;
        ++szit;
        ++ssxit;
        ++ssyit;
        ++sszit;
        ++srit;
    }
    // reduce target results to scalar
    targs.u[0][i] += vtu0.sum();
    targs.u[1][i] += vtu1.sum();
    targs.u[2][i] += vtu2.sum();
    targs.u[3][i] += vtu3.sum();
    targs.u[4][i] += vtu4.sum();
    targs.u[5][i] += vtu5.sum();
    targs.u[6][i] += vtu6.sum();
    targs.u[7][i] += vtu7.sum();
    targs.u[8][i] += vtu8.sum();
    targs.u[9][i] += vtu9.sum();
    targs.u[10][i] += vtu10.sum();
    targs.u[11][i] += vtu11.sum();

#else
    for (size_t j=jstart; j<jend; ++j) {
        nbody_kernel<S,A>(srcs.x[0][j], srcs.x[1][j], srcs.x[2][j],
                     srcs.s[0][j],  srcs.s[1][j], srcs.s[2][j], srcs.r[j],
                     targs.x[0][i], targs.x[1][i], targs.x[2][i],
                     targs.u[0][i], targs.u[1][i], targs.u[2][i],
                     targs.u[3][i], targs.u[4][i], targs.u[5][i],
                     targs.u[6][i], targs.u[7][i], targs.u[8][i],
                     targs.u[9][i], targs.u[10][i],targs.u[11][i]);
    }
#endif
}

template <class S, class A, int PD, int SD, int OD>
void ppinter(const Parts<S,A,PD,SD,OD>& __restrict__ srcs,  const size_t jstart, const size_t jend,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t istart, const size_t iend) {
    //printf("    compute srcs %ld-%ld on targs %ld-%ld\n", jstart, jend, istart, iend);

#ifdef USE_VC
    Vc::simdize<Vector<STORE>::const_iterator> sxit, syit, szit, ssxit, ssyit, sszit, srit;
    const size_t nSrcVec = (jend-jstart + Vc::Vector<S>::Size - 1) / Vc::Vector<S>::Size;

    // a simd type for A with the same number of entries as S
    typedef Vc::SimdArray<A,Vc::Vector<S>::size()> VecA;

    for (size_t i=istart; i<iend; ++i) {
        // spread this target over a vector
        const Vc::Vector<S> vtx = targs.x[0][i];
        const Vc::Vector<S> vty = targs.x[1][i];
        const Vc::Vector<S> vtz = targs.x[2][i];
        VecA vtu0(0.0f);
        VecA vtu1(0.0f);
        VecA vtu2(0.0f);
        VecA vtu3(0.0f);
        VecA vtu4(0.0f);
        VecA vtu5(0.0f);
        VecA vtu6(0.0f);
        VecA vtu7(0.0f);
        VecA vtu8(0.0f);
        VecA vtu9(0.0f);
        VecA vtu10(0.0f);
        VecA vtu11(0.0f);
        // convert source data to Vc::Vector<S>
        sxit = srcs.x[0].begin() + jstart;
        syit = srcs.x[1].begin() + jstart;
        szit = srcs.x[2].begin() + jstart;
        ssxit = srcs.s[0].begin() + jstart;
        ssyit = srcs.s[1].begin() + jstart;
        sszit = srcs.s[2].begin() + jstart;
        srit = srcs.r.begin() + jstart;
        for (size_t j=0; j<nSrcVec; ++j) {
            nbody_kernel<Vc::Vector<S>,VecA>(
                         *sxit, *syit, *szit, *ssxit, *ssyit, *sszit, *srit,
                         vtx, vty, vtz, vtu0, vtu1, vtu2, vtu3, vtu4,
                         vtu5, vtu6, vtu7, vtu8, vtu9, vtu10, vtu11);
            // advance the source iterators
            ++sxit;
            ++syit;
            ++szit;
            ++ssxit;
            ++ssyit;
            ++sszit;
            ++srit;
        }
        // reduce target results to scalar
        targs.u[0][i] += vtu0.sum();
        targs.u[1][i] += vtu1.sum();
        targs.u[2][i] += vtu2.sum();
        targs.u[3][i] += vtu3.sum();
        targs.u[4][i] += vtu4.sum();
        targs.u[5][i] += vtu5.sum();
        targs.u[6][i] += vtu6.sum();
        targs.u[7][i] += vtu7.sum();
        targs.u[8][i] += vtu8.sum();
        targs.u[9][i] += vtu9.sum();
        targs.u[10][i] += vtu10.sum();
        targs.u[11][i] += vtu11.sum();
    }
#else
    for (size_t i=istart; i<iend; ++i) {
        for (size_t j=jstart; j<jend; ++j) {
            nbody_kernel<S,A>(srcs.x[0][j], srcs.x[1][j], srcs.x[2][j],
                         srcs.s[0][j],  srcs.s[1][j], srcs.s[2][j], srcs.r[j],
                         targs.x[0][i], targs.x[1][i], targs.x[2][i],
                         targs.u[0][i], targs.u[1][i], targs.u[2][i],
                         targs.u[3][i], targs.u[4][i], targs.u[5][i],
                         targs.u[6][i], targs.u[7][i], targs.u[8][i],
                         targs.u[9][i], targs.u[10][i],targs.u[11][i]);
        }
    }
#endif
}

template <class S, int PD, int SD> class Tree;

template <class S, class A, int PD, int SD, int OD>
void tpinter(const Tree<S,PD,SD>& __restrict__ stree, const size_t j,
                   Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t i) {
    //printf("    compute srcs %ld-%ld on targ %ld\n", jstart, jend, i);
    nbody_kernel<S,A>(stree.x[0][j], stree.x[1][j], stree.x[2][j],
                 stree.s[0][j], stree.s[1][j], stree.s[2][j], stree.pr[j],
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

    static std::vector<int> test_iterations = {1, 1, 1, 1};
    bool just_build_trees = false;
    size_t numSrcs = 10000;
    size_t numTargs = 10000;
    size_t echonum = 1;
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
    Parts<STORE,ACCUM,3,3,12> srcs(numSrcs, true);
    // initialize particle data
    srcs.random_in_cube();
    //srcs.smooth_strengths();
    srcs.wave_strengths();
    //srcs.central_strengths();

    Parts<STORE,ACCUM,3,3,12> targs(numTargs, false);
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
    Tree<STORE,3,3> stree(0);
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
    Parts<STORE,ACCUM,3,3,12> eqsrcs((stree.numnodes/2) * blockSize, true);
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
    Tree<STORE,3,3> ttree(0);
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
    for (size_t i=0; i<echonum*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    for (size_t i=ntskip*((numTargs-1)/ntskip)-(echonum-1)*ntskip; i<numTargs; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
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
    for (size_t i=0; i<echonum*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    for (size_t i=ntskip*((numTargs-1)/ntskip)-(echonum-1)*ntskip; i<numTargs; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
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
    printf("error in treecode (max/rms):\t%g / %g\n", std::sqrt(maxerr/(ntskip*errcnt/(float)numTargs)), std::sqrt(errsum/errcnt));
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
    for (size_t i=0; i<echonum*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    for (size_t i=ntskip*((numTargs-1)/ntskip)-(echonum-1)*ntskip; i<numTargs; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
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
    printf("error in treecode2 (max/rms):\t%g / %g\n", std::sqrt(maxerr/(ntskip*errcnt/(float)numTargs)), std::sqrt(errsum/errcnt));
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
    for (size_t i=0; i<echonum*ntskip; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
    for (size_t i=ntskip*((numTargs-1)/ntskip)-(echonum-1)*ntskip; i<numTargs; i+=ntskip) printf("  particle %ld vel %g %g %g\n",i,targs.u[0][i],targs.u[1][i],targs.u[2][i]);
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
    printf("error in treecode3 (max/rms):\t%g / %g\n", std::sqrt(maxerr/(ntskip*errcnt/(float)numTargs)), std::sqrt(errsum/errcnt));
    }

    printf("\nDone.\n");
    return 0;
}
