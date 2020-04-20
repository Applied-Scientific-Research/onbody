/*
 * barneshut.h - parameterized Barnes-Hut O(NlogN) treecode solver
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
#include <algorithm>	// for sort and minmax
#include <numeric>	// for iota
#include <future>	// for async


// the basic unit of direct sum work is blockSize by blockSize
const size_t blockSize = 32;

//
// Find index of msb of uint32
// from http://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
//
static inline uint32_t log_2(const uint32_t x) {
    if (x == 0) return 0;
    return (31 - __builtin_clz (x));
}

//
// A set of particles, can be sources or targets
//
// templatized on (S)torage and (A)ccumulator types, and space (D)imensions
//
template <class S, class A, int D>
class Parts {
public:
    Parts(size_t);
    void resize(size_t);
    void random_in_cube();
    void smooth_strengths();
    void wave_strengths();
    void zero_vels();
    void reorder_idx(const size_t, const size_t);

    size_t n;
    // state
    alignas(32) std::array<std::vector<S>, D> x;
    // actuator (needed by sources)
    alignas(32) std::vector<S> m;
    alignas(32) std::vector<S> r;
    // results (needed by targets)
    alignas(32) std::array<std::vector<A>, D> u;
    alignas(32) std::vector<size_t> gidx;
    // temporary
    alignas(32) std::vector<size_t> lidx;
    alignas(32) std::vector<size_t> itemp;
    alignas(32) std::vector<S> ftemp;

    // useful later
    //typename S::value_type state_type;
    //typename A::value_type accumulator_type;
};

template <class S, class A, int D>
Parts<S,A,D>::Parts(size_t _num) {
    resize(_num);
}

template <class S, class A, int D>
void Parts<S,A,D>::resize(size_t _num) {
    n = _num;
    for (int d=0; d<D; ++d) x[d].resize(n);
    r.resize(n);
    m.resize(n);
    for (int d=0; d<D; ++d) u[d].resize(n);
    gidx.resize(n);
    std::iota(gidx.begin(), gidx.end(), 0);
    lidx.resize(n);
    itemp.resize(n);
    ftemp.resize(n);
}

template <class S, class A, int D>
void Parts<S,A,D>::random_in_cube() {
    for (int d=0; d<D; ++d) for (auto& _x : x[d]) { _x = (S)rand()/(S)RAND_MAX; }
    for (auto& _r : r) { _r = 1.0f / std::sqrt((S)n); }
    for (auto& _m : m) { _m = (-1.0f + 2.0f*(S)rand()/(S)RAND_MAX) / std::sqrt((S)n); }
}

template <class S, class A, int D>
void Parts<S,A,D>::smooth_strengths() {
    const S factor = 1.0 / (S)n;
    for (size_t i=0; i<n; i++) {
        m[i] = factor * (x[0][i] - x[1][i]);
    }
}

template <class S, class A, int D>
void Parts<S,A,D>::wave_strengths() {
    const S factor = 1.0 / (S)n;
    for (size_t i=0; i<n; i++) {
        const S dist = std::sqrt(std::pow(x[0][i]-0.5,2)+std::pow(x[1][i]-0.5,2));
        m[i] = factor * std::cos(30.0*std::sqrt(dist)) / (5.0*dist+1.0);
    }
}

template <class S, class A, int D>
void Parts<S,A,D>::zero_vels() {
    for (int d=0; d<D; ++d) for (auto& _u : u[d]) { _u = (S)0.0; }
}

//
// Helper function to reorder the reordering indexes
//
template <class S, class A, int D>
void Parts<S,A,D>::reorder_idx(const size_t pfirst, const size_t plast) {

    // copy the original global index vector gidx into a temporary vector
    std::copy(gidx.begin()+pfirst, gidx.begin()+plast, itemp.begin()+pfirst);

    // scatter values from the temp vector back into the original vector
    for (size_t i=pfirst; i<plast; ++i) gidx[i] = itemp[lidx[i]];
}


//
// A tree, made of a structure of arrays
//
// 0 is empty, root node is 1, children are 2,3, their children 4,5 and 6,7
// arrays always have 2^levels boxes allocated, even if some are not used
// this way, node i children are 2*i and 2*i+1
//
template <class S, int D>
class Tree {
public:
    Tree(size_t);
    void resize(size_t);
    void print(size_t);

    // number of levels in the tree
    int levels;
    // number of nodes in the tree (always 2^l)
    int numnodes;

    // tree node centers of mass
    alignas(32) std::array<std::vector<S>, D> x;
    // node size
    alignas(32) std::vector<S> s;
    // node particle radius
    alignas(32) std::vector<S> r;
    // node masses
    alignas(32) std::vector<S> m;

    // real point offset and count
    alignas(32) std::vector<size_t> ioffset;		// is this redundant?
    alignas(32) std::vector<size_t> num;
    // equivalent point offset and count
    alignas(32) std::vector<size_t> epoffset;		// is this redundant?
    alignas(32) std::vector<size_t> epnum;
};

template <class S, int D>
Tree<S,D>::Tree(size_t _num) {
    // _num is number of elements this tree needs to store
    uint32_t numLeaf = 1 + ((_num-1)/blockSize);
    //printf("  %d nodes at leaf level\n", numLeaf);
    levels = 1 + log_2(2*numLeaf-1);
    //printf("  makes %d levels in tree\n", levels);
    numnodes = 1 << levels;
    //printf("  and %d total nodes in tree\n", numnodes);
    resize(numnodes);
}

template <class S, int D>
void Tree<S,D>::resize(size_t _num) {
    numnodes = _num;
    for (int d=0; d<D; ++d) x[d].resize(numnodes);
    s.resize(numnodes);
    r.resize(numnodes);
    m.resize(numnodes);
    ioffset.resize(numnodes);
    num.resize(numnodes);
    std::fill(num.begin(), num.end(), 0);
    epoffset.resize(numnodes);
    epnum.resize(numnodes);
}

template <class S, int D>
void Tree<S,D>::print(size_t _num) {
    printf("\n%dD tree with %d levels\n", D, levels);
    for(size_t i=1; i<numnodes && i<_num; ++i) {
        printf("  %ld  %ld %ld  %g\n",i, num[i], ioffset[i], s[i]);
    }
}


//
// Caller for the O(N^2) kernel
//
template <class S, class A, int D>
float nbody_naive(const Parts<S,A,D>& __restrict__ srcs, Parts<S,A,D>& __restrict__ targs, const size_t tskip) {
    #pragma omp parallel for
    for (size_t i = 0; i < targs.n; i+=tskip) {
        (void) ppinter(srcs, 0, srcs.n, targs, i);
    }
    return (float)targs.n * (float)srcs.n * (float)nbody_kernel_flops();
}

//
// Recursive kernel for the treecode using 1st order box approximations
//
template <class S, class A, int D>
void treecode1_block(const Parts<S,A,D>& sp,
                     const Tree<S,D>& st,
                     const size_t snode,
                     Parts<S,A,D>& tp,
                     const size_t ip,
                     const float theta) {

    // if box is a leaf node, just compute the influence and return
    if (st.num[snode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        (void) ppinter(sp, st.ioffset[snode], st.ioffset[snode]+st.num[snode], tp, ip);
        return;
    }

    // distance from box center of mass to target point
    S dist = 0.0;
    for (int d=0; d<D; ++d) dist += std::pow(st.x[d][snode] - tp.x[d][ip], 2);
    dist = std::sqrt(dist);

    // is source tree node far enough away?
    if (dist / st.s[snode] > theta) {
        // box is far enough removed, approximate its influence
        nbody_kernel(st.x[0][snode], st.x[1][snode], st.r[snode], st.m[snode],
                     tp.x[0][ip], tp.x[1][ip], tp.u[0][ip], tp.u[1][ip]);
    } else {
        // box is too close, open up its children
        (void) treecode1_block<S,A,D>(sp, st, 2*snode,   tp, ip, theta);
        (void) treecode1_block<S,A,D>(sp, st, 2*snode+1, tp, ip, theta);
    }
}

//
// Caller for the simple O(NlogN) kernel
//
template <class S, class A, int D>
void nbody_treecode1(const Parts<S,A,D>& srcs, const Tree<S,D>& stree, Parts<S,A,D>& targs, const float theta) {
    #pragma omp parallel for schedule(dynamic,2*blockSize)
    for (size_t i=0; i<targs.n; ++i) {
        treecode1_block<S,A,D>(srcs, stree, (size_t)1, targs, i, theta);
    }
}

//
// Data structure for accumulating interaction counts
//
struct treecode2_stats {
    size_t sltp, sbtp;
};

//
// Recursive kernel for the treecode using equivalent particles
//
template <class S, class A, int D>
void treecode2_block(const Parts<S,A,D>& sp,
                     const Parts<S,A,D>& ep,
                     const Tree<S,D>& st,
                     const size_t snode,
                     Parts<S,A,D>& tp,
                     const size_t ip,
                     const float theta,
                     struct treecode2_stats& stats) {

    // if box is a leaf node, just compute the influence and return
    if (st.num[snode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        (void) ppinter(sp, st.ioffset[snode], st.ioffset[snode]+st.num[snode], tp, ip);
        stats.sltp++;
        return;
    }

    // distance from box center of mass to target point
    S dist = 0.0;
    for (int d=0; d<D; ++d) dist += std::pow(st.x[d][snode] - tp.x[d][ip], 2);
    dist = std::sqrt(dist) - 2.0*st.r[snode];

    // is source tree node far enough away?
    if (dist / st.s[snode] > theta) {
        // this version uses equivalent points instead!
        (void) ppinter(ep, st.epoffset[snode], st.epoffset[snode]+st.epnum[snode], tp, ip);
        stats.sbtp++;
    } else {
        // box is too close, open up its children
        (void) treecode2_block<S,A,D>(sp, ep, st, 2*snode,   tp, ip, theta, stats);
        (void) treecode2_block<S,A,D>(sp, ep, st, 2*snode+1, tp, ip, theta, stats);
    }
}

//
// Caller for the better (equivalent particle) O(NlogN) kernel
//
template <class S, class A, int D>
float nbody_treecode2(const Parts<S,A,D>& srcs,
                      const Parts<S,A,D>& eqsrcs,
                      const Tree<S,D>& stree,
                      Parts<S,A,D>& targs,
                      const float theta) {

    struct treecode2_stats stats = {0, 0};

    #pragma omp parallel
    {
        struct treecode2_stats threadstats = {0, 0};

        #pragma omp for schedule(dynamic,2*blockSize)
        for (size_t i=0; i<targs.n; ++i) {
            treecode2_block<S,A,D>(srcs, eqsrcs, stree, (size_t)1, targs, i, theta, threadstats);
        }

        #pragma omp critical
        {
            stats.sltp += threadstats.sltp;
            stats.sbtp += threadstats.sbtp;
        }
    }

    //printf("  %ld target particles averaged %g leaf-part and %g equiv-part interactions\n",
    //       targs.n, stats.sltp/(float)targs.n, stats.sbtp/(float)targs.n);
    //printf("  sltp %ld  sbtp %ld\n", stats.sltp, stats.sbtp);

    return (float)nbody_kernel_flops() * ((float)stats.sltp + (float)stats.sbtp) * (float)blockSize;
}


//
// Recursive kernel for the boxwise treecode using equivalent particles
//
template <class S, class A, int D>
void treecode3_block(const Parts<S,A,D>& sp,
                     const Parts<S,A,D>& ep,
                     const Tree<S,D>& st,
                     const size_t snode,
                     Parts<S,A,D>& tp,
                     const Tree<S,D>& tt,
                     const size_t tnode,
                     const float theta,
                     struct treecode2_stats& stats) {

    //printf("    vs src box %ld with %ld parts starting at %ld\n", snode, st.num[snode], st.ioffset[snode]);

    // if box is a leaf node, just compute the influence and return
    if (st.num[snode] <= blockSize) {
        (void) ppinter(sp, st.ioffset[snode], st.ioffset[snode]+st.num[snode],
                       tp, tt.ioffset[tnode], tt.ioffset[tnode]+tt.num[tnode]);
        stats.sltp++;
        return;
    }

    // distance from box center of mass to target point
    S dist = 0.0;
    for (int d=0; d<D; ++d) dist += std::pow(st.x[d][snode] - tt.x[d][tnode], 2);
    dist = std::sqrt(dist) - 1.0*st.r[snode] - 1.0*tt.r[tnode];

    // is source tree node far enough away?
    if (dist / (st.s[snode]+tt.s[tnode]) > theta) {
        // this version uses equivalent points instead!
        (void) ppinter(ep, st.epoffset[snode], st.epoffset[snode]+st.epnum[snode],
                       tp, tt.ioffset[tnode], tt.ioffset[tnode]+tt.num[tnode]);
        stats.sbtp++;
    } else {
        // box is too close, open up its children
        (void) treecode3_block<S,A,D>(sp, ep, st, 2*snode,   tp, tt, tnode, theta, stats);
        (void) treecode3_block<S,A,D>(sp, ep, st, 2*snode+1, tp, tt, tnode, theta, stats);
    }
}

//
// Caller for the equivalent particle O(NlogN) kernel, but interaction lists by tree
//
template <class S, class A, int D>
float nbody_treecode3(const Parts<S,A,D>& srcs,
                      const Parts<S,A,D>& eqsrcs,
                      const Tree<S,D>& stree,
                      Parts<S,A,D>& targs,
                      const Tree<S,D>& ttree,
                      const float theta) {

    struct treecode2_stats stats = {0, 0};

    #pragma omp parallel
    {
        struct treecode2_stats threadstats = {0, 0};

        #pragma omp for schedule(dynamic,8)
        for (size_t ib=0; ib<(size_t)ttree.numnodes; ++ib) {
            if (ttree.num[ib] <= blockSize and ttree.num[ib] > 0) {
                //printf("  targ box %ld has %ld parts starting at %ld\n", ib, ttree.num[ib], ttree.ioffset[ib]);
                //for (size_t ip=ttree.ioffset[ib]; ip<ttree.ioffset[ib]+ttree.num[ib]; ++ip) {
                //    for (int d=0; d<D; ++d) targs.u[d][ip] = 0.0;
                //}
                treecode3_block<S,A,D>(srcs, eqsrcs, stree, (size_t)1, targs, ttree, ib, theta, threadstats);
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
    //printf("  sltp %ld  sbtp %ld\n", stats.sltp, stats.sbtp);

    return (float)nbody_kernel_flops() * ((float)stats.sltp + (float)stats.sbtp) * (float)blockSize * (float)blockSize;
}


//
// Sort but retain only sorted index! Uses C++11 lambdas
// from http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//
template <class S>
std::vector<size_t> sortIndexes(const std::vector<S> &v) {

  // initialize original index locations
  alignas(32) std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

//
// Split into two threads, sort, and zipper
//
template <class S>
void splitSort(const int recursion_level,
               const std::vector<S> &v,
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
                        const std::vector<S> &v,
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
  //splitSort(recursion_level, v, idx, istart, istop);
}

//
// Find min and max values along an axis
//
template <class S>
std::pair<S,S> minMaxValue(const std::vector<S> &x, size_t istart, size_t iend) {

    auto itbeg = x.begin() + istart;
    auto itend = x.begin() + iend;

    auto range = std::minmax_element(itbeg, itend);
    return std::pair<S,S>(x[range.first+istart-itbeg], x[range.second+istart-itbeg]);

    // what's an initializer list?
    //return std::minmax(itbeg, itend);
}

//
// Helper function to reorder a segment of a vector
//
template <class S>
void reorder(std::vector<S> &x, std::vector<S> &t,
             const std::vector<size_t> &idx,
             const size_t pfirst, const size_t plast) {

    // copy the original input float vector x into a temporary vector
    std::copy(x.begin()+pfirst, x.begin()+plast, t.begin()+pfirst);

    // scatter values from the temp vector back into the original vector
    for (size_t i=pfirst; i<plast; ++i) x[i] = t[idx[i]];
}

//
// Make a VAMsplit k-d tree from this set of particles
// Split this segment of the particles on its longest axis
//
template <class S, class A, int D>
void splitNode(Parts<S,A,D>& p, size_t pfirst, size_t plast, Tree<S,D>& t, size_t tnode) {

    //printf("splitNode %ld  %ld %ld\n", tnode, pfirst, plast);

    const int thislev = log_2(tnode);
    #ifdef _OPENMP
    const int sort_recursion = std::max(0, (int)log_2(::omp_get_num_threads()) - thislev);
    #else
    const int sort_recursion = 0;
    #endif

    // debug print - starting condition
    //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);
    //if (pfirst == 0) printf("  splitNode at level ? with %n threads\n", ::omp_get_num_threads());
    #ifdef _OPENMP
    //if (pfirst == 0) printf("  splitNode at level %d with %d threads, %d recursions\n", thislev, ::omp_get_num_threads(), sort_recursion);
    #else
    //if (pfirst == 0) printf("  splitNode at level %d with 1 threads, %d recursions\n", thislev, sort_recursion);
    #endif

    // find the min/max of the three axes
    std::array<S,D> boxsizes;
    for (int d=0; d<D; ++d) {
        auto minmax = minMaxValue(p.x[d], pfirst, plast);
        boxsizes[d] = minmax.second - minmax.first;
    }

    // find total mass and center of mass - old way
    // copy strength vector
    alignas(32) decltype(p.m) absstr(p.m.begin()+pfirst, p.m.begin()+plast);
    // find abs() of each entry using a lambda
    std::for_each(absstr.begin(), absstr.end(), [](float &str){ str = std::abs(str); });

    // sum of abs of strengths
    S ooass = (S)1.0 / std::accumulate(absstr.begin(), absstr.end(), 0.0);

    for (int d=0; d<D; ++d) {
        t.x[d][tnode] = ooass * std::inner_product(p.x[d].begin()+pfirst, p.x[d].begin()+plast, absstr.begin(), 0.0);
    }
    //printf("  abs mass %g and cm %g %g\n", t.m[tnode], t.x[tnode], t.y[tnode]);

    // sum of vectorial strengths
    t.m[tnode] = std::accumulate(p.m.begin()+pfirst, p.m.begin()+plast, 0.0);

    // fine average particle radius
    S radsum = std::accumulate(p.r.begin()+pfirst, p.r.begin()+plast, 0.0);
    t.r[tnode] = radsum / (S)(plast-pfirst);

    // new way: compute the sum of the absolute values of the point "masses" - slower!
    //t.m[tnode] = 0.0;
    //t.x[tnode] = 0.0;
    //t.y[tnode] = 0.0;
    //for (size_t i=pfirst; i<plast; ++i) {
    //    const S thisabs = std::abs(p.m[i]);
    //    t.m[tnode] += thisabs;
    //    t.x[tnode] += p.x[i] * thisabs;
    //    t.y[tnode] += p.y[i] * thisabs;
    //}
    //t.x[tnode] /= t.m[tnode];
    //t.y[tnode] /= t.m[tnode];

    // write all this data to the tree node
    t.ioffset[tnode] = pfirst;
    t.num[tnode] = plast - pfirst;
    //printf("  tree node has offset %d and num %d\n", t.ioffset[tnode], t.num[tnode]);

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();
    //printf("  longest axis is %ld, length %g\n", maxaxis, boxsizes[maxaxis]);
    S bsss = 0.0;
    for (int d=0; d<D; ++d) bsss += std::pow(boxsizes[d],2);
    t.s[tnode] = 0.5 * std::sqrt(bsss);
    //printf("  tree node time:\t[%.3f] million cycles\n", get_elapsed_mcycles());
    //printf("  box %ld size %g and rad %g\n", tnode, t.s[tnode], t.r[tnode]);

    // no need to split or compute further
    if (t.num[tnode] <= blockSize) {
        // we are at block size!
        //printf("  tree node %ld position %g %g size %g %g\n", tnode, t.x[tnode], t.y[tnode], boxsizes[0], boxsizes[1]);
        return;
    }

    // sort this portion of the array along the big axis
    //printf("sort\n");
    //if (pfirst == 0) reset_and_start_timer();
    (void) sortIndexesSection(sort_recursion, p.x[maxaxis], p.lidx, pfirst, plast);
    //if (pfirst == 0) printf("    sort time:\t\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // rearrange the elements - parallel sections did not make things faster
    //printf("reorder\n");
    for (int d=0; d<D; ++d) reorder(p.x[d], p.ftemp, p.lidx, pfirst, plast);
    reorder(p.m, p.ftemp, p.lidx, pfirst, plast);
    reorder(p.r, p.ftemp, p.lidx, pfirst, plast);
    p.reorder_idx(pfirst, plast);
    //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);
    //if (pfirst == 0) printf("    reorder time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // determine where the split should be
    size_t pmiddle = pfirst + blockSize * (1 << log_2((t.num[tnode]-1)/blockSize));
    //printf("split at %ld %ld %ld into nodes %d %d\n", pfirst, pmiddle, plast, 2*tnode, 2*tnode+1);

    // recursively call this routine for this node's new children
    #pragma omp task shared(p,t)
    (void) splitNode(p, pfirst,  pmiddle, t, 2*tnode);
    #pragma omp task shared(p,t)
    (void) splitNode(p, pmiddle, plast,   t, 2*tnode+1);

    // we don't need a taskwait directive here, because we don't use an upward pass, though it may
    //   seem like that would be a better way to compute the tree node's mass and cm, it's slower

    if (tnode == 1) {
        // this is executed on the final call
    }
}

//
// Recursively refine leaf node's particles until they are hierarchically nearby
// Code is borrowed from splitNode above
//
template <class S, class A, int D>
void refineLeaf(Parts<S,A,D>& p, Tree<S,D>& t, size_t pfirst, size_t plast) {

    // if there are 1 or 2 particles, then they are already in "order"
    if (plast-pfirst < 3) return;

    // perform very much the same action as tree-build
    //printf("    refining particles %ld to %ld\n", pfirst, plast);

    // find the min/max of the three axes
    std::array<S,D> boxsizes;
    for (int d=0; d<D; ++d) {
        auto minmax = minMaxValue(p.x[d], pfirst, plast);
        boxsizes[d] = minmax.second - minmax.first;
    }

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();

    // sort this portion of the array along the big axis
    (void) sortIndexesSection(0, p.x[maxaxis], p.lidx, pfirst, plast);

    // rearrange the elements
    for (int d=0; d<D; ++d) reorder(p.x[d], p.ftemp, p.lidx, pfirst, plast);
    reorder(p.m, p.ftemp, p.lidx, pfirst, plast);
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
template <class S, class A, int D>
void refineTree(Parts<S,A,D>& p, Tree<S,D>& t, size_t tnode) {
    //printf("  node %d has %d particles\n", tnode, t.num[tnode]);
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
}

//
// Loop over all nodes in the tree and merge adjacent pairs of particles
//
// one situation: we are a leaf node
//       another: we are a non-leaf node taking eq parts from two leaf nodes
//       another: we are a non-leaf node taking eq parts from two non-leaf nodes
//       another: we are a non-leaf node taking eq parts from one leaf and one non-leaf node
//
template <class S, class A, int D>
void calcEquivalents(Parts<S,A,D>& p, Parts<S,A,D>& ep, Tree<S,D>& t, size_t tnode) {
    //printf("  node %d has %d particles\n", tnode, t.num[tnode]);

    t.epoffset[tnode] = tnode * blockSize;
    t.epnum[tnode] = 0;
    //printf("    equivalent particles start at %d\n", t.epoffset[tnode]);

    // loop over children, adding equivalent particles to our list
    for (size_t ichild = 2*tnode; ichild < 2*tnode+2; ++ichild) {
        //printf("  child %d has %d particles\n", ichild, t.num[ichild]);

        // split on whether this child is a leaf node or not
        if (t.num[ichild] > blockSize) {
            // this child is a non-leaf node and needs to make equivalent particles
            (void) calcEquivalents(p, ep, t, ichild);

            //printf("  back in node %d...\n", tnode);

            // now we read those equivalent particles and make higher-level equivalents
            //printf("    child %d made equiv parts %d to %d\n", ichild, t.epoffset[ichild], t.epoffset[ichild]+t.epnum[ichild]);

            // merge pairs of child's equivalent particles until we have half
            size_t numEqps = (t.epnum[ichild]+1) / 2;
            size_t istart = (blockSize/2) * ichild;
            size_t istop = istart + numEqps;
            //printf("    making %d equivalent particles %d to %d\n", numEqps, istart, istop);

            // loop over new equivalent particles and real particles together
            size_t iep = istart;
            size_t ip = t.epoffset[ichild] + 1;
            for (; iep<istop and ip<t.epoffset[ichild]+t.epnum[ichild];
                   iep++,     ip+=2) {
                //printf("    merging %d and %d into %d\n", ip-1,ip,iep);
                const S str1 = std::max((S)1.e-20, std::abs(ep.m[ip-1]));
                const S str2 = std::max((S)1.e-20, std::abs(ep.m[ip]));
                const S pairm = 1.0 / (str1 + str2);
                for (int d=0; d<D; ++d) ep.x[d][iep] = (ep.x[d][ip-1]*str1 + ep.x[d][ip]*str2) * pairm;
                ep.r[iep] = std::sqrt((std::pow(ep.r[ip-1],2)*str1 + std::pow(ep.r[ip],2)*str2) * pairm);
                ep.m[iep] = ep.m[ip-1] + ep.m[ip];
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.epoffset[ichild]+t.epnum[ichild]) {
                //printf("    passing %d up into %d\n", ip-1,iep);
                for (int d=0; d<D; ++d) ep.x[d][iep] = ep.x[d][ip-1];
                ep.r[iep] = ep.r[ip-1];
                ep.m[iep] = ep.m[ip-1];
            }
            t.epnum[tnode] += numEqps;
        } else {
            // this child is a leaf node
            //printf("    child leaf node has particles %d to %d\n", t.ioffset[ichild], t.ioffset[ichild]+t.num[ichild]);

            // if we're a leaf node, merge pairs of particles until we have half
            size_t numEqps = (t.num[ichild]+1) / 2;
            size_t istart = (blockSize/2) * ichild;
            size_t istop = istart + numEqps;
            //printf("    making %d equivalent particles %d to %d\n", numEqps, istart, istop);

            // loop over new equivalent particles and real particles together
            size_t iep = istart;
            size_t ip = t.ioffset[ichild] + 1;
            for (; iep<istop and ip<t.ioffset[ichild]+t.num[ichild];
                   iep++,     ip+=2) {
                //printf("    merging %d and %d into %d\n", ip-1,ip,iep);
                const S str1 = std::max((S)1.e-20, std::abs(p.m[ip-1]));
                const S str2 = std::max((S)1.e-20, std::abs(p.m[ip]));
                const S pairm = 1.0 / (str1 + str2);
                for (int d=0; d<D; ++d) ep.x[d][iep] = (p.x[d][ip-1]*str1 + p.x[d][ip]*str2) * pairm;
                ep.r[iep] = std::sqrt((std::pow(p.r[ip-1],2)*str1 + std::pow(p.r[ip],2)*str2) * pairm);
                ep.m[iep] = p.m[ip-1] + p.m[ip];
                //if (ep.r[iep] != ep.r[iep]) {
                //    printf("nan detected at ep %ld\n", iep);
                //    printf("  pos %g %g and %g %g\n", p.x[0][ip-1], p.x[1][ip-1], p.x[0][ip], p.x[1][ip]);
                //    printf("  str %g %g and rads %g %g\n", p.m[ip-1], p.m[ip], p.r[ip-1], p.r[ip]);
                //}
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.ioffset[ichild]+t.num[ichild]) {
                //printf("    passing %d up into %d\n", ip-1,iep);
                for (int d=0; d<D; ++d) ep.x[d][iep] = p.x[d][ip-1];
                ep.r[iep] = p.r[ip-1];
                ep.m[iep] = p.m[ip-1];
            }
            t.epnum[tnode] += numEqps;
        }
    }

    //printf("  node %d finally has %d equivalent particles, offset %d\n", tnode, t.epnum[tnode], t.epoffset[tnode]);
}

