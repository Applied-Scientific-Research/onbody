/*
 * barneshut.h - parameterized Barnes-Hut O(NlogN) treecode solver
 *
 * Copyright (c) 2017-22, Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#include "Parts.hpp"
#include "Tree.hpp"

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
#include <algorithm>	// for sort and minmax
#include <numeric>	// for iota
#include <future>	// for async

// use a fast partial sort (select) algorithm for tree-building
#define PARTIAL_SORT

#ifdef USE_VC
template <class S> using Vector = std::vector<S, Vc::Allocator<S>>;
template <class S> size_t VecSize = Vc::Vector<S>::Size;
#else
template <class S> using Vector = std::vector<S>;
template <class S> size_t VecSize = 1;
#endif


//
// Caller for the O(N^2) kernel
//
template <class S, class A, int PD, int SD, int OD>
float nbody_naive(const Parts<S,A,PD,SD,OD>& __restrict__ srcs, Parts<S,A,PD,SD,OD>& __restrict__ targs, const size_t tskip) {
    #pragma omp parallel for
    for (size_t i=0; i<targs.n; i+=tskip) {
        (void) ppinter(srcs, 0, srcs.n, targs, i);
    }
    return (float)(targs.n/tskip) * (float)srcs.n * (float)nbody_kernel_flops();
}

//
// Data structure for accumulating interaction counts
//
struct treecode_stats {
    size_t sltp, sbtp;
};

//
// Recursive kernel for the treecode using 1st order box approximations
//
template <class S, class A, int PD, int SD, int OD>
void treecode1_block(const Parts<S,A,PD,SD,OD>& sp,
                     const Tree<S,PD,SD>& st,
                     const size_t snode,
                     Parts<S,A,PD,SD,OD>& tp,
                     const size_t ip,
                     const float theta,
                     struct treecode_stats& stats) {

    // if box is a leaf node, just compute the influence and return
    if (st.num[snode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        (void) ppinter(sp, st.ioffset[snode], st.ioffset[snode]+st.num[snode], tp, ip);
        stats.sltp++;
        return;
    }

    // distance from box center of mass to target point
    S dist = 0.0;
    //for (int d=0; d<PD; ++d) dist += std::pow(st.x[d][snode] - tp.x[d][ip], 2);
    for (int d=0; d<PD; ++d) dist += std::pow(std::max((S)0.0, std::abs(st.x[d][snode] - tp.x[d][ip]) - (S)0.5*st.ns[d][snode]), 2);
    dist = std::sqrt(dist);
    // scale the distance in the box-opening criterion?
    //if (PD == 2) dist = std::exp(0.75*std::log(dist));
    //else dist = std::exp(0.666667*std::log(dist));

    // is source tree node far enough away?
    //if (dist / st.nr[snode] > theta) {
    if (dist / (2.0*st.nr[snode]) > theta) {
        // box is far enough removed, approximate its influence
        (void) tpinter(st, snode, tp, ip);
        stats.sbtp++;
    } else {
        // box is too close, open up its children
        (void) treecode1_block<S,A,PD,SD,OD>(sp, st, 2*snode,   tp, ip, theta, stats);
        (void) treecode1_block<S,A,PD,SD,OD>(sp, st, 2*snode+1, tp, ip, theta, stats);
    }
}

//
// Caller for the simple O(NlogN) kernel
//
template <class S, class A, int PD, int SD, int OD>
float nbody_treecode1(const Parts<S,A,PD,SD,OD>& srcs,
                      const Tree<S,PD,SD>& stree,
                      Parts<S,A,PD,SD,OD>& targs,
                      const float theta) {

    struct treecode_stats stats = {0, 0};

    #pragma omp parallel
    {
        struct treecode_stats threadstats = {0, 0};

        #pragma omp for schedule(dynamic,2*blockSize)
        for (size_t i=0; i<targs.n; ++i) {
            treecode1_block<S,A,PD,SD,OD>(srcs, stree, (size_t)1, targs, i, theta, threadstats);
        }

        #pragma omp critical
        {
            stats.sltp += threadstats.sltp;
            stats.sbtp += threadstats.sbtp;
        }
    }

    return (float)nbody_kernel_flops() * ((float)stats.sbtp + (float)stats.sltp*(float)blockSize);
}

//
// Recursive kernel for the treecode using equivalent particles
//
template <class S, class A, int PD, int SD, int OD>
void treecode2_block(const Parts<S,A,PD,SD,OD>& sp,
                     const Parts<S,A,PD,SD,OD>& ep,
                     const Tree<S,PD,SD>& st,
                     const size_t snode,
                     Parts<S,A,PD,SD,OD>& tp,
                     const size_t ip,
                     const float theta,
                     struct treecode_stats& stats) {

    // if box is a leaf node, just compute the influence and return
    if (st.num[snode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        (void) ppinter(sp, st.ioffset[snode], st.ioffset[snode]+st.num[snode], tp, ip);
        stats.sltp++;
        return;
    }

    // distance from box center of mass to target point
    S dist = 0.0;

    const bool use_traditional_mac = true;

    if (use_traditional_mac) {
        for (int d=0; d<PD; ++d) dist += std::pow(st.nc[d][snode] - tp.x[d][ip], 2);
        dist = std::sqrt(dist);

    } else {
        for (int d=0; d<PD; ++d) dist += std::pow(std::max((S)0.0, std::abs(st.x[d][snode] - tp.x[d][ip]) - (S)0.5*st.ns[d][snode]), 2);
        //dist = std::sqrt(dist) - 2.0*st.pr[snode];
        dist = std::sqrt(dist);
        // scale the distance in the box-opening criterion?
        if (PD == 2) dist = std::exp(0.75*std::log(dist));
        else dist = std::exp(0.666667*std::log(dist));
    }

    // is source tree node far enough away?
    //if (dist / st.nr[snode] > theta) {
    if (dist / (2.0*st.nr[snode]) > theta) {
        // this version uses equivalent points instead!
        (void) ppinter(ep, st.epoffset[snode], st.epoffset[snode]+st.epnum[snode], tp, ip);
        stats.sbtp++;
    } else {
        // box is too close, open up its children
        (void) treecode2_block<S,A,PD,SD,OD>(sp, ep, st, 2*snode,   tp, ip, theta, stats);
        (void) treecode2_block<S,A,PD,SD,OD>(sp, ep, st, 2*snode+1, tp, ip, theta, stats);
    }
}

//
// Caller for the better (equivalent particle) O(NlogN) kernel
//
template <class S, class A, int PD, int SD, int OD>
float nbody_treecode2(const Parts<S,A,PD,SD,OD>& srcs,
                      const Parts<S,A,PD,SD,OD>& eqsrcs,
                      const Tree<S,PD,SD>& stree,
                      Parts<S,A,PD,SD,OD>& targs,
                      const float theta) {

    struct treecode_stats stats = {0, 0};

    #pragma omp parallel
    {
        struct treecode_stats threadstats = {0, 0};

        // this is 1-3% slower
        //#pragma omp for schedule(dynamic,blockSize/2)
        #pragma omp for schedule(dynamic,2*blockSize)
        for (size_t i=0; i<targs.n; ++i) {
            treecode2_block<S,A,PD,SD,OD>(srcs, eqsrcs, stree, (size_t)1, targs, i, theta, threadstats);
        }

        #pragma omp critical
        {
            stats.sltp += threadstats.sltp;
            stats.sbtp += threadstats.sbtp;
        }
    }

    //printf("  %ld target particles averaged %g leaf-part and %g equiv-part interactions\n",
    //       targs.n, stats.sltp/(float)targs.n, stats.sbtp/(float)targs.n);
    //printf("  sltp %ld  sbtp %ld  epnum %ld\n", stats.sltp, stats.sbtp, stree.epnum[1]);

    return (float)nbody_kernel_flops() * 
           ((float)stats.sltp*(float)blockSize + (float)stats.sbtp*(float)stree.epnum[1]);
}


//
// Recursive kernel for the boxwise treecode using equivalent particles
//
template <class S, class A, int PD, int SD, int OD>
void treecode3_block(const Parts<S,A,PD,SD,OD>& sp,
                     const Parts<S,A,PD,SD,OD>& ep,
                     const Tree<S,PD,SD>& st,
                     const size_t snode,
                     Parts<S,A,PD,SD,OD>& tp,
                     const Tree<S,PD,SD>& tt,
                     const size_t tnode,
                     const float theta,
                     struct treecode_stats& stats) {

    //printf("    vs src box %ld with %ld parts starting at %ld\n", snode, st.num[snode], st.ioffset[snode]);

    // if box is a leaf node, just compute the influence and return
    if (st.num[snode] <= blockSize) {
        (void) ppinter(sp, st.ioffset[snode], st.ioffset[snode]+st.num[snode],
                       tp, tt.ioffset[tnode], tt.ioffset[tnode]+tt.num[tnode]);
        stats.sltp++;
        return;
    }

    S dist = 0.0;

    const bool use_traditional_mac = true;

    if (use_traditional_mac) {
        for (int d=0; d<PD; ++d) dist += std::pow(st.nc[d][snode] - tt.nc[d][tnode], 2);
        dist = std::sqrt(dist);

    } else {

        // minimum distance between box corners - THIS IS WRONG!!! x is not the center but the cm!
        for (int d=0; d<PD; ++d) dist += std::pow(std::max((S)0.0, std::abs(st.x[d][snode] - tt.x[d][tnode]) - (S)0.5*(st.ns[d][snode]+tt.ns[d][tnode])), 2);
        dist = std::sqrt(dist);

        // include the box's mean particle size? no, no real benefit
        //dist += - 1.0*st.pr[snode] - 1.0*tt.pr[tnode];
        //dist += + 1.0*st.pr[snode] + 1.0*tt.pr[tnode];

        // scale the distance in the box-opening criterion?
        if (PD == 2) dist = std::exp(0.75*std::log(dist));
        else dist = std::exp(0.75*std::log(dist));
        //else dist = std::exp(0.666667*std::log(dist));
        //dist = std::exp(0.75*std::log(dist));
        //dist = std::exp(0.666667*std::log(dist));
        //dist = std::sqrt(dist);
    }

    // is source tree node far enough away?
    //if (dist / (st.nr[snode]+tt.nr[tnode]) > theta) {
    if (dist / (2.0*st.nr[snode]) > theta) {
        // this version uses equivalent points instead!
        (void) ppinter(ep, st.epoffset[snode], st.epoffset[snode]+st.epnum[snode],
                       tp, tt.ioffset[tnode], tt.ioffset[tnode]+tt.num[tnode]);
        stats.sbtp++;
    } else {
        // box is too close, open up its children
        (void) treecode3_block<S,A,PD,SD,OD>(sp, ep, st, 2*snode,   tp, tt, tnode, theta, stats);
        (void) treecode3_block<S,A,PD,SD,OD>(sp, ep, st, 2*snode+1, tp, tt, tnode, theta, stats);
    }
}

//
// Caller for the equivalent particle O(NlogN) kernel, but interaction lists by tree
//
template <class S, class A, int PD, int SD, int OD>
float nbody_treecode3(const Parts<S,A,PD,SD,OD>& srcs,
                      const Parts<S,A,PD,SD,OD>& eqsrcs,
                      const Tree<S,PD,SD>& stree,
                      Parts<S,A,PD,SD,OD>& targs,
                      const Tree<S,PD,SD>& ttree,
                      const float theta) {

    struct treecode_stats stats = {0, 0};

    #pragma omp parallel
    {
        struct treecode_stats threadstats = {0, 0};

        #pragma omp for schedule(dynamic,8)
        for (size_t ib=0; ib<(size_t)ttree.numnodes; ++ib) {
            if (ttree.num[ib] <= blockSize and ttree.num[ib] > 0) {
                //printf("  targ box %ld has %ld parts starting at %ld\n", ib, ttree.num[ib], ttree.ioffset[ib]);
                //for (size_t ip=ttree.ioffset[ib]; ip<ttree.ioffset[ib]+ttree.num[ib]; ++ip) {
                //    for (int d=0; d<OD; ++d) targs.u[d][ip] = 0.0;
                //}
                treecode3_block<S,A,PD,SD,OD>(srcs, eqsrcs, stree, (size_t)1, targs, ttree, ib, theta, threadstats);
            }
        }

        #pragma omp critical
        {
            stats.sltp += threadstats.sltp;
            stats.sbtp += threadstats.sbtp;
        }
    }

    //printf("  %ld target particles averaged %g leaf-part and %g equiv-part interactions\n",
    //       targs.n, stats.sltp/(float)targs.n, stats.sbtp/(float)targs.n);
    //printf("  sltp %ld  sbtp %ld  epnum %ld\n", stats.sltp, stats.sbtp, stree.epnum[1]);

    return (float)nbody_kernel_flops() * (float)blockSize *
           ((float)stats.sltp*(float)blockSize + (float)stats.sbtp*(float)stree.epnum[1]);
}


//
// Split into two threads, sort, and zipper
//
template <class S>
void splitSort(const int recursion_level,
               const Vector<S> &v,
               std::vector<size_t> &idx,
               const size_t istart, const size_t istop) {

  //printf("      inside splitSort with level %d\n", recursion_level);
  // std::inplace_merge() is your friend here! as is recursion


  if (recursion_level == 0 or (istop-istart < 100) ) {
    // we are parallel enough: let this thread sort its chunk
    //printf("      performing standard sort\n");

    // sort indexes based on comparing values in v
    std::sort(idx.begin()+istart,
              idx.begin()+istop,
              [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  } else {
    // we are not parallel enough: fork and join until we are
    //printf("      performing split sort\n");

    const size_t imiddle = istart + (istop-istart)/2;

    // fork a thread for the first half
    #ifdef _OPENMP
    auto handle = std::async(std::launch::async,
                             splitSort<S>, recursion_level-1, std::cref(v), std::ref(idx), istart, imiddle);
    #else
    splitSort(recursion_level-1, v, idx, istart, imiddle);
    #endif

    // do the second half here
    splitSort(recursion_level-1, v, idx, imiddle, istop);

    // force the thread to join
    #ifdef _OPENMP
    handle.get();
    #endif

    // zipper the results together
    std::inplace_merge(idx.begin()+istart,
                       idx.begin()+imiddle,
                       idx.begin()+istop,
                       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  }

}

//
// Sort but retain only sorted index! Uses C++11 lambdas
// from http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//
template <class S>
void sortIndexesSection(const int recursion_level,
                        const Vector<S> &v,
                        std::vector<size_t> &idx,
                        const size_t istart, const size_t istop) {

  // use omp to figure out how many threads are being used now,
  // then split into threads to perform the sort in parallel, then
  // zipper them all together again

  // initialize original index locations
  std::iota(idx.begin()+istart, idx.begin()+istop, istart);

  // sort indexes based on comparing values in v
  std::sort(idx.begin()+istart, idx.begin()+istop,
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  // sort indexes based on comparing values in v, possibly with forking
  if (false) {
    splitSort(recursion_level, v, idx, istart, istop);
  }
}

//
// Find min and max values along an axis
//
template <class S>
std::pair<S,S> minMaxValue(const Vector<S> &x, const size_t istart, const size_t iend) {

    auto range = std::minmax_element(x.begin() + istart, x.begin() + iend);
    return std::pair<S,S>(x[range.first-x.begin()], x[range.second-x.begin()]);

    // what's an initializer list?
    //return std::minmax(itbeg, itend);
}

// do it in parallel
template <class S>
std::pair<S,S> minMaxValue(const Vector<S> &x, const size_t istart, const size_t iend, const int nthreads) {

    S xmin = 9.9e+9;
    S xmax = -9.9e+9;

    #pragma omp taskloop if(nthreads>1) shared(x) reduction(min:xmin) reduction(max:xmax)
    for (int i=0; i<nthreads; ++i) {
        const size_t ifirst = istart + i*(iend-istart)/nthreads;
        const size_t ilast = istart + (i+1)*(iend-istart)/nthreads;
        auto range = std::minmax_element(x.begin()+ifirst, x.begin()+ilast);
        S thisxmin = x[range.first-x.begin()];
        S thisxmax = x[range.second-x.begin()];
        // need to compare to xmin! 
        if (thisxmin < xmin) xmin = thisxmin;
        if (thisxmax > xmax) xmax = thisxmax;
        //std::cout << "chunk " << i << " from " << ifirst << " to " << ilast << " has min " << (range.first-x.begin()) << " " << thisxmin << " max " << (range.second-x.begin()) << " " << thisxmax << std::endl;
    }

    return std::pair<S,S>(xmin, xmax);
}

// do it in parallel, slowly
template <class S>
std::pair<S,S> minMaxValue_slow(const Vector<S> &x, const size_t istart, const size_t iend, const int nthreads) {

    S xmin = 9.9e+9;
    S xmax = -9.9e+9;

    //#pragma omp for num_threads(nthreads) reduction(min:xmin) reduction(max:xmax)
    for (size_t i=istart; i<iend; ++i) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
    }

    return std::pair<S,S>(xmin, xmax);
}

//
// Helper function to reorder a segment of a vector
//
template <class S>
void reorder(Vector<S>& x, Vector<S>& t,
             const std::vector<size_t>& idx,
             const size_t pfirst, const size_t plast) {

    // copy the original input float vector x into a temporary vector
    std::copy(x.begin()+pfirst, x.begin()+plast, t.begin()+pfirst);

    // scatter values from the temp vector back into the original vector
    for (size_t i=pfirst; i<plast; ++i) x[i] = t[idx[i]];
}

// only barely faster
template <class S>
void reorder(Vector<S>& x, Vector<S>& t,
             const std::vector<size_t>& idx,
             const size_t pfirst, const size_t plast,
             const int nthreads) {

    // copy the original input float vector x into a temporary vector
    std::copy(x.begin()+pfirst, x.begin()+plast, t.begin()+pfirst);

    // scatter values from the temp vector back into the original vector
    #pragma omp taskloop if(nthreads>1) num_tasks(nthreads) shared(x,t,idx)
    for (size_t i=pfirst; i<plast; ++i) x[i] = t[idx[i]];
}

//
// Perform an experimental partial sort
//
template <class S>
void partialSortIndexes(Vector<S>& v, std::vector<size_t>& idx, const std::pair<S,S> in_minmax,
                        const size_t istart, const size_t nless, const size_t istop) {

  assert(v.size() == idx.size() && "Array size mismatch in partialSortIndexes");
  assert(v.size() >= istop && "Not enough data in partialSortIndexes");
  assert(istop > istart && "Invalid sort range in partialSortIndexes");
  assert(nless > istart && "Invalid cutoff in partialSortIndexes");
  assert(nless < istop && "Invalid cutoff in partialSortIndexes");

  // initialize original index locations
  std::iota(idx.begin()+istart, idx.begin()+istop, istart);

  // set initial low and high indicies for the test area
  size_t pfirst = istart;
  size_t plast = istop-1;
  int iters = 0;
  S cutoff_ideal = (S)(nless-istart) / (S)(istop-istart);

  // use passed-in min/max as first range bounds
  std::pair<S,S> minmax = in_minmax;

  while (plast > pfirst and iters < 100) {

    if (iters > 0) {
      //auto start = std::chrono::steady_clock::now();
      minmax = minMaxValue(v, pfirst, plast+1);
      //auto end = std::chrono::steady_clock::now();
      //std::chrono::duration<double> elapsed_seconds = end-start;
      //if (plast-pfirst>10000000) printf("        sort minmax:\t[%.4f] seconds\n", elapsed_seconds.count());
    }

    // estimate value at cutoff
    S frac = (S)(nless-0.5-pfirst) / (S)(plast-pfirst);
    frac = (9.0*frac + 1.0*cutoff_ideal) / 10.0;
    S testval = minmax.first + (minmax.second-minmax.first) * frac;

    // march from both ends looking for pairs to swap
    size_t tfirst = pfirst;
    size_t tlast = plast;

    while (true) {
      while (v[tfirst] < testval) ++tfirst;
      while (v[tlast] >= testval) --tlast;
      if (tlast > tfirst) {
        // why not use std::swap? try it. go ahead. i'll wait here.
        const S val = v[tfirst];
        v[tfirst] = v[tlast];
        v[tlast] = val;
        const size_t ival = idx[tfirst];
        idx[tfirst] = idx[tlast];
        idx[tlast] = ival;
      } else {
        break;
      }
    }

    // now all points from pfirst to tlast are below, 
    // and all points from tfirst to plast are above

    const size_t oldpfirst = pfirst;
    const size_t oldplast = plast;

    // test for completion, or refine bounds for next step
    if (tfirst == nless) {
      // sort complete!
      break;
    } else if (tfirst < nless) {
      // reset lower bound
      pfirst = tfirst;
    } else {
      // rest upper bound
      plast = tlast;
    }

    // if pfirst and plast didn't change, then the middle numbers are all the same
    if (pfirst == oldpfirst and plast == oldplast) {
      break;
    }

    iters++;
  }
}


//
// Make a VAMsplit k-d tree from this set of particles
// Split this segment of the particles on its longest axis
//
template <class S, class A, int PD, int SD, int OD>
void splitNode(Parts<S,A,PD,SD,OD>& p, const size_t pfirst, const size_t plast, Tree<S,PD,SD>& t, const size_t tnode) {

    //printf("splitNode %ld  %ld %ld\n", tnode, pfirst, plast);

    // must know how many threads we are allowed to play with
    #ifdef _OPENMP
    const int thislev = log_2(tnode);
    const int sort_recursion = std::max(0, (int)log_2(::omp_get_num_threads()) - thislev);
    #else
    const int sort_recursion = 0;
    #endif
    const int threads_per_node = std::pow(2, sort_recursion);

    // debug print - starting condition
    //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);
    //if (pfirst == 0) printf("  splitNode at level ? with %n threads\n", ::omp_get_num_threads());
    #ifdef _OPENMP
    //if (pfirst == 0) printf("  splitNode at level %d with %d threads, %d recursions\n", thislev, ::omp_get_num_threads(), sort_recursion);
    #else
    //if (pfirst == 0) printf("  splitNode at level %d with 1 threads, %d recursions\n", thislev, sort_recursion);
    #endif

    auto start = std::chrono::steady_clock::now();

    // find the min/max of the three axes and save them
    std::array<std::pair<S,S>,PD> boxbounds;
    for (int d=0; d<PD; ++d) {
        boxbounds[d] = minMaxValue(p.x[d], pfirst, plast, threads_per_node);
        t.ns[d][tnode] = boxbounds[d].second - boxbounds[d].first;
        t.nc[d][tnode] = 0.5 * (boxbounds[d].second + boxbounds[d].first);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    //if (tnode==1) printf("      1st minmax:\t[%.4f] seconds\n", elapsed_seconds.count());

    // write particle data to the tree node
    t.ioffset[tnode] = pfirst;
    t.num[tnode] = plast - pfirst;
    //printf("  tree node has offset %d and num %d\n", t.ioffset[tnode], t.num[tnode]);

    // find box/node radius
    S bsss = 0.0;
    for (int d=0; d<PD; ++d) bsss += std::pow(t.ns[d][tnode],2);
    t.nr[tnode] = 0.5 * std::sqrt(bsss);
    //printf("  tree node time:\t[%.3f] million cycles\n", get_elapsed_mcycles());
    //printf("  box %ld size %g and rad %g\n", tnode, t.s[tnode], t.r[tnode]);

    // no need to split or compute further
    if (t.num[tnode] <= blockSize) {
        // we are at block size!
        //printf("  tree node %ld position %g %g size %g %g\n", tnode, t.x[tnode], t.y[tnode], boxsizes[0], boxsizes[1]);
        return;
    }

    // find longest box edge
    //auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();
    int maxaxis = 0;
    S maxaxissize = -1.0;
    for (int d=0; d<PD; ++d) {
        if (t.ns[d][tnode] > maxaxissize) {
            maxaxissize = t.ns[d][tnode];
            maxaxis = d;
        }
    }
    //printf("  longest axis is %ld, length %g\n", maxaxis, boxsizes[maxaxis]);

    // determine where the split should be
    size_t pmiddle = pfirst + blockSize * (1 << log_2((t.num[tnode]-1)/blockSize));
    //printf("split at %ld %ld %ld into nodes %d %d\n", pfirst, pmiddle, plast, 2*tnode, 2*tnode+1);

    // temporary list of vectors to be sorted - oh, then we'll need many new temp vectors
    //std::vector<Vector<S>&> jumble;

    start = std::chrono::steady_clock::now();

    // sort this portion of the array along the big axis
#ifdef PARTIAL_SORT
    (void) partialSortIndexes(p.x[maxaxis], p.lidx, boxbounds[maxaxis], pfirst, pmiddle, plast);

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start;
    //if (tnode==1) printf("      1st partial sort:\t[%.4f] seconds\n", elapsed_seconds.count());

    start = std::chrono::steady_clock::now();
    // note, this also sorts the values - don't parallelize, we share ftemp!
    //#pragma omp taskloop if(threads_per_node>1) shared(p)
    for (int d=0; d<PD; ++d) {
        if (d != maxaxis) reorder(p.x[d], p.ftemp, p.lidx, pfirst, plast, threads_per_node);
    }
#else
    (void) sortIndexesSection(sort_recursion, p.x[maxaxis], p.lidx, pfirst, plast);
    for (int d=0; d<PD; ++d) reorder(p.x[d], p.ftemp, p.lidx, pfirst, plast, threads_per_node);
#endif


    // rearrange the elements - parallel sections did not make things faster
    if (p.are_sources) for (int d=0; d<SD; ++d) reorder(p.s[d], p.ftemp, p.lidx, pfirst, plast, threads_per_node);
    reorder(p.r, p.ftemp, p.lidx, pfirst, plast, threads_per_node);
    p.reorder_idx(pfirst, plast);

    end = std::chrono::steady_clock::now();
    elapsed_seconds = end-start;
    //if (tnode==1) printf("      1st reorder:\t[%.4f] seconds\n", elapsed_seconds.count());

    // recursively call this routine for this node's new children, but only spawn new tasks if we have some to spare
    #pragma omp task if(threads_per_node>1) shared(p,t)
    (void) splitNode(p, pfirst,  pmiddle, t, 2*tnode);
    #pragma omp task if(threads_per_node>1) shared(p,t)
    (void) splitNode(p, pmiddle, plast,   t, 2*tnode+1);

    // we don't need a taskwait directive here, because we don't use an upward pass, though it may
    //   seem like that would be a better way to compute the tree node's mass and cm, it's slower

    if (tnode == 1) {
        // this is executed on the final call
    }
}

//
// Finish up the tree with a downward pass
//
template <class S, class A, int PD, int SD, int OD>
void finishTree(Parts<S,A,PD,SD,OD>& p, Tree<S,PD,SD>& t, const size_t tnode) {

    // if we're not a leaf node...
    if (t.num[tnode] > blockSize) {
        const size_t child1 = 2*tnode;
        const size_t child2 = 2*tnode+1;

        // recursively call this routine for this node's children
        #pragma omp task shared(p,t)
        (void) finishTree(p, t, child1);
        #pragma omp task shared(p,t)
        (void) finishTree(p, t, child2);
        #pragma omp taskwait

        // once both children are done, we can merge their data for the parent
        const S oonp = (S)1.0 / (t.num[child1] + t.num[child2]);

        // first, the center of "mass"
        for (int d=0; d<PD; ++d) {
            t.x[d][tnode] = oonp * (t.num[child1]*t.x[d][child1] + t.num[child2]*t.x[d][child2]);
        }

        // then the vectorial strength of the node
        for (int d=0; d<SD; ++d) {
            t.s[d][tnode] = t.s[d][child1] + t.s[d][child2];
        }

        // lastly, the radius
        t.pr[tnode] = oonp * (t.num[child1]*t.pr[child1] + t.num[child2]*t.pr[child2]);

        //printf("box %ld has cm %g %g %g str %g %g %g and rad %g\n", tnode, t.x[0][tnode], t.x[1][tnode], t.x[2][tnode], t.s[0][tnode], t.s[1][tnode], t.s[2][tnode], t.pr[tnode]);

        // total flops: 6 + PD*4 + SD*1

    // else we are a leaf node, then compute the box weights, etc.
    } else {

        const size_t pfirst = t.ioffset[tnode];
        const size_t plast = pfirst + t.num[tnode];

        // find total mass and center of mass - old way
        alignas(32) Vector<S> absstr(plast-pfirst);

        if (p.are_sources) {
        if (SD == 1) {
            // find abs() of each entry using a lambda
            absstr = Vector<S>(p.s[0].begin()+pfirst, p.s[0].begin()+plast);
            std::for_each(absstr.begin(), absstr.end(), [](S &str){ str = std::abs(str); });
            //std::fill(absstr.begin(), absstr.end(), (S)0.0);
            //for (size_t i=pfirst; i<plast; ++i) {
            //    absstr[i-pfirst] = std::abs(p.s[0][i]);
            //}
        } else {
            // find abs() of each entry with a loop and sqrt
            std::fill(absstr.begin(), absstr.end(), (S)0.0);
            for (int d=0; d<SD; ++d) {
                for (size_t i=pfirst; i<plast; ++i) {
                    absstr[i-pfirst] += std::pow(p.s[d][i], 2);
                }
            }
            std::for_each(absstr.begin(), absstr.end(), [](S &str){ str = std::sqrt(str); });
        }
        } else {
            // Parts are targets
            std::fill(absstr.begin(), absstr.end(), (S)1.0);
        }

        // one over the sum of abs of strengths
        const S ooass = (S)1.0 / (1.e-20 + std::accumulate(absstr.begin(), absstr.end(), 0.0));

        // compute the node center of "mass"
        for (int d=0; d<PD; ++d) {
            t.x[d][tnode] = ooass * std::inner_product(p.x[d].begin()+pfirst, p.x[d].begin()+plast, absstr.begin(), 0.0);
        }
        //printf("  abs mass %g and cm %g %g\n", t.s[0][tnode], t.x[tnode], t.y[tnode]);

        // sum of vectorial strengths
        if (p.are_sources) for (int d=0; d<SD; ++d) {
            t.s[d][tnode] = std::accumulate(p.s[d].begin()+pfirst, p.s[d].begin()+plast, 0.0);
        }

        // fine average particle radius
        const S radsum = std::accumulate(p.r.begin()+pfirst, p.r.begin()+plast, 0.0);
        t.pr[tnode] = radsum / (S)t.num[tnode];

        // total flops: t.num[tnode] * (1+1+SD+2*PD+2*SD+1) or (3+2*PD+3*SD)

    // end computation for leaf node
    }
}

//
// Make a VAMsplit k-d tree from this set of particles
// This uses two passes: one upward to sort and build the tree
// The second downward to assign properties to the tree
//
template <class S, class A, int PD, int SD, int OD>
void makeTree(Parts<S,A,PD,SD,OD>& p, Tree<S,PD,SD>& t) {

    // allocate temporaries
    auto start = std::chrono::steady_clock::now();
    p.lidx.resize(p.n);
    p.itemp.resize(p.n);
    p.ftemp.resize(p.n);
    p.gidx.resize(p.n);
    std::iota(p.gidx.begin(), p.gidx.end(), 0);

    // allocate
    t = Tree<S,PD,SD>(p.n);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    //printf("    tree allocation:\t[%.4f] seconds\n", elapsed_seconds.count());

    // upward pass, starting at node 1 (root) and recursing
    start = std::chrono::steady_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) splitNode(p, 0, p.n, t, 1);
    #pragma omp taskwait
    end = std::chrono::steady_clock::now(); elapsed_seconds = end-start;
    //printf("    tree upward pass:\t[%.4f] seconds\n", elapsed_seconds.count());

    // downward pass, calculate masses, etc.
    start = std::chrono::steady_clock::now();
    #pragma omp parallel
    #pragma omp single
    (void) finishTree(p, t, 1);
    #pragma omp taskwait
    end = std::chrono::steady_clock::now(); elapsed_seconds = end-start;
    //printf("    tree dwnwrd pass:\t[%.4f] seconds\n", elapsed_seconds.count());

    // de-allocate temporaries
    p.lidx.resize(0);
    p.itemp.resize(0);
    p.ftemp.resize(0);
    if (p.are_sources) p.gidx.resize(0);
}

//
// Recursively refine leaf node's particles until they are hierarchically nearby
// Code is borrowed from splitNode above
//
template <class S, class A, int PD, int SD, int OD>
void refineLeaf(Parts<S,A,PD,SD,OD>& p, Tree<S,PD,SD> const & t, const size_t pfirst, const size_t plast) {

    // if there are 1 or 2 particles, then they are already in "order"
    if (plast-pfirst < 3) return;

    // perform very much the same action as tree-build
    //printf("    refining particles %ld to %ld\n", pfirst, plast);

    // find the min/max of the three axes
    std::array<S,PD> boxsizes;
    for (int d=0; d<PD; ++d) {
        //auto minmax = minMaxValue(p.x[d], pfirst, plast, 1);
        auto minmax = minMaxValue(p.x[d], pfirst, plast);
        boxsizes[d] = minmax.second - minmax.first;
    }

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();

    // sort this portion of the array along the big axis
    (void) sortIndexesSection(0, p.x[maxaxis], p.lidx, pfirst, plast);

    // rearrange the elements
    for (int d=0; d<PD; ++d) reorder(p.x[d], p.ftemp, p.lidx, pfirst, plast);
    if (p.are_sources) for (int d=0; d<SD; ++d) reorder(p.s[d], p.ftemp, p.lidx, pfirst, plast);
    reorder(p.r, p.ftemp, p.lidx, pfirst, plast);
    p.reorder_idx(pfirst, plast);

    // determine where the split should be
    size_t pmiddle = pfirst + (1 << log_2(plast-pfirst-1));

    // recursively call this routine for this node's new children
    (void) refineLeaf(p, t, pfirst,  pmiddle);
    (void) refineLeaf(p, t, pmiddle, plast);
}

//
// Loop over all leaf nodes in the tree and call the refine function on them
//
template <class S, class A, int PD, int SD, int OD>
void refineTree(Parts<S,A,PD,SD,OD>& p, Tree<S,PD,SD> const & t, const size_t tnode) {

    if (tnode == 1) {
        // allocate temporaries
        p.lidx.resize(p.n);
        p.itemp.resize(p.n);
        p.ftemp.resize(p.n);
        if (p.are_sources) {
            p.gidx.resize(p.n);
            std::iota(p.gidx.begin(), p.gidx.end(), 0);
        }
    }

    //printf("  node %ld has %ld particles\n", tnode, t.num[tnode]);
    if (t.num[tnode] <= blockSize) {
        // make the equivalent particles for this node
        (void) refineLeaf(p, t, t.ioffset[tnode], t.ioffset[tnode]+t.num[tnode]);
        //for (size_t i=t.ioffset[tnode]; i<t.ioffset[tnode]+t.num[tnode]; ++i)
        //    printf("  %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);
    } else {
        // recurse and check child nodes
        #pragma omp task shared(p,t)
        (void) refineTree(p, t, 2*tnode);
        #pragma omp task shared(p,t)
        (void) refineTree(p, t, 2*tnode+1);
    }

    if (tnode == 1) {
        #pragma omp taskwait
        // de-allocate temporaries
        p.lidx.resize(0);
        p.itemp.resize(0);
        p.ftemp.resize(0);
        if (p.are_sources) p.gidx.resize(0);
    }
}

//
// Loop over all nodes in the tree and merge adjacent pairs of particles
//
// one situation: we are a leaf node
//       another: we are a non-leaf node taking eq parts from two leaf nodes
//       another: we are a non-leaf node taking eq parts from two non-leaf nodes
//       another: we are a non-leaf node taking eq parts from one leaf and one non-leaf node
//
template <class S, class A, int PD, int SD, int OD>
void calcEquivalents(Parts<S,A,PD,SD,OD> const & p,
                     Parts<S,A,PD,SD,OD>& ep,
                     Tree<S,PD,SD>& t,
                     const size_t tnode) {

    //printf("  node %d has %d particles\n", tnode, t.num[tnode]);
    if (not p.are_sources or not ep.are_sources) return;

    t.epoffset[tnode] = tnode * blockSize;
    t.epnum[tnode] = 0;
    //printf("    equivalent particles start at %ld\n", t.epoffset[tnode]);

    // loop over children, adding equivalent particles to our list
    for (size_t ichild = 2*tnode; ichild < 2*tnode+2; ++ichild) {
        //printf("  child %ld has %ld particles\n", ichild, t.num[ichild]);

        // split on whether this child is a leaf node or not
        if (t.num[ichild] > blockSize) {
            // this child is a non-leaf node and needs to make equivalent particles
            (void) calcEquivalents(p, ep, t, ichild);

            //printf("  back in node %d...\n", tnode);

            // now we read those equivalent particles and make higher-level equivalents
            //printf("    child %ld made equiv parts %ld to %ld\n", ichild, t.epoffset[ichild], t.epoffset[ichild]+t.epnum[ichild]);

            // merge pairs of child's equivalent particles until we have half
            size_t numEqps = (t.epnum[ichild]+1) / 2;
            size_t istart = (blockSize/2) * ichild;
            size_t istop = istart + numEqps;
            //printf("    making %ld equivalent particles %ld to %ld\n", numEqps, istart, istop);

            // loop over new equivalent particles and real particles together
            size_t iep = istart;
            size_t ip = t.epoffset[ichild] + 1;
            for (; iep<istop and ip<t.epoffset[ichild]+t.epnum[ichild];
                   iep++,     ip+=2) {
                //printf("    merging %ld and %ld into %ld\n", ip-1,ip,iep);
                S str1, str2;
                if (SD == 1) {
                    str1 = std::max((S)1.e-20, std::abs(ep.s[0][ip-1]));
                    str2 = std::max((S)1.e-20, std::abs(ep.s[0][ip]));
                } else {
                    str1 = (S)0.0;
                    for (int d=0; d<SD; ++d) str1 += std::pow(ep.s[d][ip-1], 2);
                    str1 = std::max((S)1.e-20, std::sqrt(str1));
                    str2 = (S)0.0;
                    for (int d=0; d<SD; ++d) str2 += std::pow(ep.s[d][ip], 2);
                    str2 = std::max((S)1.e-20, std::sqrt(str2));
                }
                const S pairm = 1.0 / (str1 + str2);
                for (int d=0; d<PD; ++d) ep.x[d][iep] = (ep.x[d][ip-1]*str1 + ep.x[d][ip]*str2) * pairm;
                ep.r[iep] = std::sqrt((std::pow(ep.r[ip-1],2)*str1 + std::pow(ep.r[ip],2)*str2) * pairm);
                for (int d=0; d<SD; ++d) ep.s[d][iep] = ep.s[d][ip-1] + ep.s[d][ip];
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.epoffset[ichild]+t.epnum[ichild]) {
                //printf("    passing %ld up into %ld\n", ip-1,iep);
                for (int d=0; d<PD; ++d) ep.x[d][iep] = ep.x[d][ip-1];
                for (int d=0; d<SD; ++d) ep.s[d][iep] = ep.s[d][ip-1];
                ep.r[iep] = ep.r[ip-1];
            }
            t.epnum[tnode] += numEqps;

        } else {
            // this child is a leaf node
            //printf("    child leaf node has particles %ld to %ld\n", t.ioffset[ichild], t.ioffset[ichild]+t.num[ichild]);

            // if we're a leaf node, merge pairs of particles until we have half
            size_t numEqps = (t.num[ichild]+1) / 2;
            size_t istart = (blockSize/2) * ichild;
            size_t istop = istart + numEqps;
            //printf("    making %ld equivalent particles %ld to %ld\n", numEqps, istart, istop);

            // loop over new equivalent particles and real particles together
            size_t iep = istart;
            size_t ip = t.ioffset[ichild] + 1;
            for (; iep<istop and ip<t.ioffset[ichild]+t.num[ichild];
                   iep++,     ip+=2) {
                //printf("    merging %ld and %ld into %ld\n", ip-1,ip,iep);
                S str1, str2;
                if (SD == 1) {
                    str1 = std::max((S)1.e-20, std::abs(p.s[0][ip-1]));
                    str2 = std::max((S)1.e-20, std::abs(p.s[0][ip]));
                } else {
                    str1 = (S)0.0;
                    for (int d=0; d<SD; ++d) str1 += std::pow(p.s[d][ip-1], 2);
                    str1 = std::max((S)1.e-20, std::sqrt(str1));
                    str2 = (S)0.0;
                    for (int d=0; d<SD; ++d) str2 += std::pow(p.s[d][ip], 2);
                    str2 = std::max((S)1.e-20, std::sqrt(str2));
                }
                const S pairm = 1.0 / (str1 + str2);
                for (int d=0; d<PD; ++d) ep.x[d][iep] = (p.x[d][ip-1]*str1 + p.x[d][ip]*str2) * pairm;
                ep.r[iep] = std::sqrt((std::pow(p.r[ip-1],2)*str1 + std::pow(p.r[ip],2)*str2) * pairm);
                for (int d=0; d<SD; ++d) ep.s[d][iep] = p.s[d][ip-1] + p.s[d][ip];
                //if (ep.r[iep] != ep.r[iep]) {
                //    printf("nan detected at ep %ld\n", iep);
                //    printf("  pos %g %g and %g %g\n", p.x[0][ip-1], p.x[1][ip-1], p.x[0][ip], p.x[1][ip]);
                //    printf("  str %g %g and rads %g %g\n", p.s[0][ip-1], p.s[0][ip], p.r[ip-1], p.r[ip]);
                //}
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.ioffset[ichild]+t.num[ichild]) {
                //printf("    passing %ld up into %ld\n", ip-1,iep);
                for (int d=0; d<PD; ++d) ep.x[d][iep] = p.x[d][ip-1];
                ep.r[iep] = p.r[ip-1];
                for (int d=0; d<SD; ++d) ep.s[d][iep] = p.s[d][ip-1];
            }
            t.epnum[tnode] += numEqps;
        }
    }

    //printf("  node %d finally has %d equivalent particles, offset %d\n", tnode, t.epnum[tnode], t.epoffset[tnode]);
}

