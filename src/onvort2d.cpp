/*
 * onvort2d - testbed for an O(N) 2d vortex solver
 *
 * Copyright (c) 2017, Mark J Stock
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
#include <algorithm>	// for sort and minmax
#include <numeric>	// for iota
#include <future>	// for async

#include "barneshut.h"
//#include "Polynomial.hh"
#include "wlspoly.hpp"

const char* progname = "onvort2d";

//
// Approximate a spatial derivative from a number of irregularly-spaced points
//
template <class S, class A>
A least_squares_val(const S xt, const S yt,
                    const std::vector<S>& x, const std::vector<S>& y,
                    const std::vector<A>& u,
                    const size_t istart, const size_t iend) {

    //printf("  target point at %g %g %g\n", xt, yt, zt);

    // prepare the arrays for CxxPolyFit
    std::vector<S> xs(2*(iend-istart));
    std::vector<S> vs(iend-istart);

    // fill the arrays
    size_t icnt = 0;
    for (size_t i=0; i<iend-istart; ++i) {
        // eventually want weighted least squares
        //const S dist = std::sqrt(dx*dx+dy*dy);
        // we should really use radius to scale this weight!!!
        //const S weight = 1.f / (0.001f + dist);
        size_t idx = istart+i;

        // but for now, keep it simple
        xs[icnt++] = x[idx] - xt;
        xs[icnt++] = y[idx] - yt;
        vs[i] = u[idx];
    }

    // generate the least squares fit (not weighted yet)
    // template params are type, dimensions, polynomial order
    WLSPoly<S,2,1> lsfit;
    lsfit.solve(xs, vs);

    // evaluate at xt,yt, the origin
    std::vector<S> xep = {0.0, 0.0};
    return (S)lsfit.eval(xep);
}

//
// Data structure for accumulating interaction counts
//
struct fastsumm_stats {
    size_t sltl, sbtl, sltb, sbtb, tlc, lpc, bpc;
};

//
// Caller for the fast summation O(N) method
//
// ittn is the target tree node that this routine will work on
// itsv is the source tree node vector that will affect ittn
//
// We will change u,v,w for the targs points and the eqtargs equivalent points
//
template <class S, class A, int D>
struct fastsumm_stats nbody_fastsumm(const Parts<S,A,D>& srcs, const Parts<S,A,D>& eqsrcs, const Tree<S,D>& stree,
                    Parts<S,A,D>& targs, Parts<S,A,D>& eqtargs, const Tree<S,D>& ttree,
                    const size_t ittn, std::vector<size_t> istv_in, const float theta) {

    // start counters
    struct fastsumm_stats stats = {0, 0, 0, 0, 0, 0, 0};

    // quit out if there are no particles in this box
    if (ttree.num[ittn] < 1) return stats;

    //printf("Targ box %d is affected by %lu source boxes at this level\n",ittn,istv.size());
    const bool targetIsLeaf = ttree.num[ittn] <= blockSize;

    // prepare the target arrays for accumulations
    if (targetIsLeaf) {
        stats.tlc++;
        // zero the velocities
        std::fill_n(&(targs.u[ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);
        std::fill_n(&(targs.v[ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);

        if (ittn > 1) {
            // prolongation operation: take the parent's equiv points and move any
            // velocity from those to our real points
            const size_t destStart = ttree.ioffset[ittn];
            const size_t destNum = ttree.num[ittn];
            const size_t origStart = ttree.epoffset[ittn/2] + (blockSize/2) * (ittn%2);
            //const size_t origNum = (destNum+1)/2;
            //printf("  copying parent equiv parts %d to %d to our own real parts %d to %d\n",
            //       origStart, origStart+origNum, destStart, destStart+destNum);
            for (size_t i=0; i<destNum; ++i) {
                const size_t idest = destStart + i;
                const size_t iorig = origStart + i/2;
                //printf("    %d at %g %g %g is parent of %d at %g %g %g\n",
                //       iorig, eqtargs.x[iorig], eqtargs.y[iorig], eqtargs.z[iorig],
                //       idest,   targs.x[idest],   targs.y[idest],   targs.z[idest]);
                // second take, use linear least squares to approximate value
                if (true) {
                    const size_t nearest = 16;
                    const size_t istart = nearest*(iorig/nearest);
                    const size_t iend = istart+nearest;
                    //printf("  approximating velocity at equiv pt %d from equiv pt %d\n", idest, iorig);
                    targs.u[idest] = least_squares_val(targs.x[idest], targs.y[idest],
                                                       eqtargs.x, eqtargs.y, eqtargs.u, istart, iend);
                    targs.v[idest] = least_squares_val(targs.x[idest], targs.y[idest],
                                                       eqtargs.x, eqtargs.y, eqtargs.v, istart, iend);
                } else {
                    // as a first take, simply copy the result to the children
                    targs.u[idest] = eqtargs.u[iorig];
                    targs.v[idest] = eqtargs.v[iorig];
                }
            }
            stats.lpc++;
        }

    } else {
        // zero the equivalent particle velocities
        std::fill_n(&(eqtargs.u[ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);
        std::fill_n(&(eqtargs.v[ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);

        if (ittn > 1) {
            // prolongation operation: take the parent's equiv points and move any
            // velocity from those to our equiv points
            const size_t destStart = ttree.epoffset[ittn];
            const size_t destNum = ttree.epnum[ittn];
            const size_t origStart = ttree.epoffset[ittn/2] + (blockSize/2) * (ittn%2);
            //const size_t origNum = (destNum+1)/2;
            //printf("  copying parent equiv parts %d to %d to our own equiv parts %d to %d\n",
            //       origStart, origStart+origNum, destStart, destStart+destNum);

            //for (size_t i=0; i<(ttree.epnum[ittn]+1)/2; ++i) {
            //    size_t ipe = ttree.epoffset[ittn]/2 + i;
            //    printf("    %d  %g %g %g\n", ipe, eqtargs.u[ipe], eqtargs.v[ipe], eqtargs.w[ipe]);
            //}

            for (size_t i=0; i<destNum; ++i) {
                const size_t idest = destStart + i;
                const size_t iorig = origStart + i/2;
                //printf("    %d at %g %g %g is parent of %d at %g %g %g\n",
                //       iorig, eqtargs.x[iorig], eqtargs.y[iorig], eqtargs.z[iorig],
                //       idest, eqtargs.x[idest], eqtargs.y[idest], eqtargs.z[idest]);
                // second take, apply gradient of value to delta location
                if (true) {
                    const size_t nearest = 16;
                    const size_t istart = nearest*(iorig/nearest);
                    const size_t iend = istart+nearest;
                    //printf("  approximating velocity at equiv pt %d from equiv pt %d\n", idest, iorig);
                    eqtargs.u[idest] = least_squares_val(eqtargs.x[idest], eqtargs.y[idest],
                                                         eqtargs.x, eqtargs.y, eqtargs.u, istart, iend);
                    eqtargs.v[idest] = least_squares_val(eqtargs.x[idest], eqtargs.y[idest],
                                                         eqtargs.x, eqtargs.y, eqtargs.v, istart, iend);
                } else {
                    // as a first take, simply copy the result to the children
                    eqtargs.u[idest] = eqtargs.u[iorig];
                    eqtargs.v[idest] = eqtargs.v[iorig];
                }
            }
            stats.bpc++;
        }
    }

    // initialize a new vector of source boxes to pass to this target box's children
    std::vector<size_t> cstv;

    // make a local copy of the input source tree vector
    std::vector<size_t> istv = istv_in;

    // for target box ittn, check all unaccounted-for source boxes
    size_t num_istv = istv.size();
    for (size_t i=0; i<num_istv; i++) {
        const size_t sn = istv[i];

        // skip this loop iteration
        if (stree.num[sn] < 1) continue;

        const bool sourceIsLeaf = stree.num[sn] <= blockSize;
        //printf("  source %d affects target %d\n",sn,ittn);

        // if source box is a leaf node, just compute the influence and return?
        // this assumes target box is also a leaf node!
        if (sourceIsLeaf and targetIsLeaf) {
            //printf("    real on real, srcs %d to %d, targs %d to %d\n", stree.ioffset[sn], stree.ioffset[sn]   + stree.num[sn], ttree.ioffset[ittn], ttree.ioffset[ittn] + ttree.num[ittn]);

            // compute all-on-all direct influence
            for (size_t i = ttree.ioffset[ittn]; i < ttree.ioffset[ittn] + ttree.num[ittn]; i++) {
            for (size_t j = stree.ioffset[sn];   j < stree.ioffset[sn]   + stree.num[sn];   j++) {
                nbody_kernel(srcs.x[j],  srcs.y[j], srcs.r[j], srcs.m[j],
                             targs.x[i], targs.y[i],
                             targs.u[i], targs.v[i]);
            }
            }
            stats.sltl++;
            continue;
        }

        // distance from box center of mass to target point
        const S dx = stree.x[sn] - ttree.x[ittn];
        const S dy = stree.y[sn] - ttree.y[ittn];
        const S diag = stree.s[sn] + ttree.s[ittn];
        const S dist = std::sqrt(dx*dx + dy*dy);
        //printf("  src box %d is %g away and diag %g\n",sn, dist, diag);

        // split on what to do with this pair
        if (dist / diag > theta) {
            // it is far enough - we can approximate
            //printf("    well-separated\n");

            if (sourceIsLeaf) {
                // compute real source particles on equivalent target points
                for (size_t i = ttree.epoffset[ittn]; i < ttree.epoffset[ittn] + ttree.epnum[ittn]; i++) {
                for (size_t j = stree.ioffset[sn];    j < stree.ioffset[sn]    + stree.num[sn];     j++) {
                    nbody_kernel(srcs.x[j],    srcs.y[j],  srcs.r[j], srcs.m[j],
                                 eqtargs.x[i], eqtargs.y[i],
                                 eqtargs.u[i], eqtargs.v[i]);
                }
                }
                stats.sltb++;

            } else if (targetIsLeaf) {
                // compute equivalent source particles on real target points
                for (size_t i = ttree.ioffset[ittn]; i < ttree.ioffset[ittn] + ttree.num[ittn]; i++) {
                for (size_t j = stree.epoffset[sn];  j < stree.epoffset[sn]  + stree.epnum[sn]; j++) {
                    nbody_kernel(eqsrcs.x[j], eqsrcs.y[j], eqsrcs.r[j], eqsrcs.m[j],
                                 targs.x[i],  targs.y[i],
                                 targs.u[i],  targs.v[i]);
                }
                }
                stats.sbtl++;

            } else {
                // compute equivalent source particles on equivalent target points
                for (size_t i = ttree.epoffset[ittn]; i < ttree.epoffset[ittn] + ttree.epnum[ittn]; i++) {
                for (size_t j = stree.epoffset[sn];   j < stree.epoffset[sn]   + stree.epnum[sn];   j++) {
                    nbody_kernel(eqsrcs.x[j],  eqsrcs.y[j],  eqsrcs.r[j], eqsrcs.m[j],
                                 eqtargs.x[i], eqtargs.y[i],
                                 eqtargs.u[i], eqtargs.v[i]);
                }
                }
                stats.sbtb++;
            }

        } else if (ttree.s[ittn] > 0.7*stree.s[sn]) {
        //} else if (true) {
            // target box is larger than source box; try to refine targets first
            //printf("    not well-separated, target is larger\n");

            if (targetIsLeaf) {
                // this means source is NOT leaf
                // put children of source box onto the end of the current list
                istv.push_back(2*sn);
                istv.push_back(2*sn+1);
                num_istv += 2;
                //printf("    pushing %d and %d to the end of this list\n", 2*sn, 2*sn+1);
            } else {
                // put this source box on the new list for target's children
                cstv.push_back(sn);
                //printf("    pushing %d to the end of the new list\n", sn);
            }

        } else {
            // source box is larger than target box; try to refine sources first
            //printf("    not well-separated, source is larger\n");

            if (sourceIsLeaf) {
                // this means target is NOT leaf
                // put this source box on the new list for target's children
                cstv.push_back(sn);
                //printf("    pushing %d to the end of the new list\n", sn);
            } else {
                // put children of source box onto the end of the current list
                istv.push_back(2*sn);
                istv.push_back(2*sn+1);
                num_istv += 2;
                //printf("    pushing %d and %d to the end of this list\n", 2*sn, 2*sn+1);
            }
        }
        //printf("    istv now has %lu entries\n",istv.size());
    }

    if (targetIsLeaf) {
        //printf("  leaf box %ld  sltl %ld  sbtl %ld\n", ittn, stats.sltl, stats.sbtl);

    } else {
        //printf("  non-leaf box %ld                     sltb %ld  sbtb %ld\n", ittn, stats.sltb, stats.sbtb);
        // prolongation of equivalent particle velocities to children's equivalent particles

        // recurse onto the target box's children
        struct fastsumm_stats cstats1, cstats2;

        #pragma omp task shared(srcs,eqsrcs,stree,targs,eqtargs,ttree,cstats1)
        cstats1 = nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree, 2*ittn, cstv, theta);

        #pragma omp task shared(srcs,eqsrcs,stree,targs,eqtargs,ttree,cstats2)
        cstats2 = nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree, 2*ittn+1, cstv, theta);

        // accumulate the child box's stats - but must wait until preceding tasks complete
        #pragma omp taskwait
        stats.sltl += cstats1.sltl + cstats2.sltl;
        stats.sbtl += cstats1.sbtl + cstats2.sbtl;
        stats.sltb += cstats1.sltb + cstats2.sltb;
        stats.sbtb += cstats1.sbtb + cstats2.sbtb;
        stats.tlc  += cstats1.tlc  + cstats2.tlc;
        stats.lpc  += cstats1.lpc  + cstats2.lpc;
        stats.bpc  += cstats1.bpc  + cstats2.bpc;
    }

    // report counter results
    if (ittn == 1) {
        #pragma omp taskwait
        printf("  %ld target leaf nodes averaged %g leaf-leaf and %g equiv-leaf interactions\n",
               stats.tlc, stats.sltl/(float)stats.tlc, stats.sbtl/(float)stats.tlc);
        printf("  sltl %ld  sbtl %ld  sltb %ld  sbtb %ld\n", stats.sltl, stats.sbtl, stats.sltb, stats.sbtb);
        printf("  leaf prolongation count %ld  box pc %ld\n", stats.lpc, stats.bpc);
    }

    //printf("  box %ld  sltl %ld  sbtl %ld  sltb %ld  sbtb %ld\n", ittn, stats.sltl, stats.sbtl, stats.sltb, stats.sbtb);
    return stats;
}

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

    static std::vector<int> test_iterations = {1, 0, 1, 0};
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
    Parts<float,double,2> srcs(numSrcs);
    // initialize particle data
    srcs.random_in_cube();
    //srcs.smooth_strengths();
    srcs.wave_strengths();

    Parts<float,double,2> targs(numTargs);
    // initialize particle data
    targs.random_in_cube();
    for (auto& m : targs.m) { m = 1.0f; }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("  init parts time:\t\t[%.4f] seconds\n", elapsed_seconds.count());


    // allocate and initialize tree
    printf("\nBuilding the source tree\n");
    start = std::chrono::system_clock::now();
    Tree<float,2> stree(numSrcs);
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
    Parts<float,double,2> eqsrcs((stree.numnodes/2) * blockSize);
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


    // don't need the target tree for treecode, but will for fast code
    printf("\nBuilding the target tree\n");
    start = std::chrono::system_clock::now();
    Tree<float,2> ttree(numTargs);
    printf("  with %ld particles and block size of %ld\n", numTargs, blockSize);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  allocate and init tree:\t[%.4f] seconds\n", elapsed_seconds.count());

    // split this node and recurse
    start = std::chrono::system_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) splitNode(targs, 0, targs.n, ttree, 1);
    #pragma omp taskwait
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  build tree time:\t\t[%.4f] seconds\n", elapsed_seconds.count());

    //ttree.print(300);

    // find equivalent points
    printf("\nCalculating equivalent targ points\n");
    start = std::chrono::system_clock::now();
    Parts<float,double,2> eqtargs((ttree.numnodes/2) * blockSize);
    printf("  need %ld particles\n", eqtargs.n);
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  allocate eqtargs structures:\t[%.4f] seconds\n", elapsed_seconds.count());

    // first, reorder tree until all parts are adjacent in space-filling curve
    start = std::chrono::system_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) refineTree(targs, ttree, 1);
    #pragma omp taskwait
    end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
    printf("  refine within leaf nodes:\t[%.4f] seconds\n", elapsed_seconds.count());

    // then, march through arrays merging pairs as you go up
    start = std::chrono::system_clock::now();
    (void) calcEquivalents(targs, eqtargs, ttree, 1);
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
        nbody_treecode2(srcs, eqsrcs, stree, targs, 4.0f);
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


    //
    // Run the new O(N) equivalent particle method
    //
    if (test_iterations[3] > 0) {
    printf("\nRun the fast O(N) method\n");
    double minFast = 1e30;
    for (int i = 0; i < test_iterations[3]; ++i) {
        start = std::chrono::system_clock::now();
        std::vector<size_t> source_boxes = {1};
        // theta=0.82f roughly matches treecode2's 1.4f re: num of leaf-leaf interactions
        // theta=1.5f roughly matches treecode2's 1.4f re: RMS error
        #pragma omp parallel
        #pragma omp single
        (void) nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree,
                              1, source_boxes, 1.0f);
        #pragma omp taskwait
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minFast = std::min(minFast, dt);
    }
    printf("[onbody fast]:\t\t\t[%.4f] seconds\n", minFast);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g\n",i,targs.u[i],targs.v[i]);
    // save the results for comparison
    std::vector<float> fastu(targs.u.begin(), targs.u.end());

    // compare accuracy
    errsum = 0.0;
    errcnt = 0.0;
    for (size_t i=0; i< targs.u.size(); i+=ntskip) {
        float thiserr = fastu[i]-naiveu[i];
        errsum += thiserr*thiserr;
        errcnt += naiveu[i]*naiveu[i];
    }
    printf("RMS error in fastsumm is %g\n", std::sqrt(errsum/errcnt));
    }

    return 0;
}
