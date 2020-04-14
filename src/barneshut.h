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
const size_t blockSize = 64;

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
// templatized on storage and accumulator types
//
template <class S, class A>
class Parts {
public:
    Parts(size_t);
    void resize(size_t);

    size_t n;
    // state
    alignas(32) std::vector<S> x;
    alignas(32) std::vector<S> y;
    alignas(32) std::vector<S> r;
    // actuator (needed by sources)
    alignas(32) std::vector<S> m;
    // results (needed by targets)
    alignas(32) std::vector<A> u;
    alignas(32) std::vector<A> v;
    // temporary
    alignas(32) std::vector<size_t> itemp;
    alignas(32) std::vector<S> ftemp;

    // useful later
    //typename S::value_type state_type;
    //typename A::value_type accumulator_type;
};

template <class S, class A>
Parts<S,A>::Parts(size_t _num) {
    resize(_num);
}

template <class S, class A>
void Parts<S,A>::resize(size_t _num) {
    n = _num;
    x.resize(n);
    y.resize(n);
    r.resize(n);
    m.resize(n);
    u.resize(n);
    v.resize(n);
    itemp.resize(n);
    ftemp.resize(n);
}


//
// A tree, made of a structure of arrays
//
// 0 is empty, root node is 1, children are 2,3, their children 4,5 and 6,7
// arrays always have 2^levels boxes allocated, even if some are not used
// this way, node i children are 2*i and 2*i+1
//
template <class S>
class Tree {
public:
    Tree(size_t);
    void resize(size_t);
    void print(size_t);

    // number of levels in the tree
    int levels;
    // number of nodes in the tree (always 2^l)
    int numnodes;

    // tree node centers (of mass?)
    alignas(32) std::vector<S> x;
    alignas(32) std::vector<S> y;
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

template <class S>
Tree<S>::Tree(size_t _num) {
    // _num is number of elements this tree needs to store
    uint32_t numLeaf = 1 + ((_num-1)/blockSize);
    printf("  %d nodes at leaf level\n", numLeaf);
    levels = 1 + log_2(2*numLeaf-1);
    printf("  makes %d levels in tree\n", levels);
    numnodes = 1 << levels;
    printf("  and %d total nodes in tree\n", numnodes);
    resize(numnodes);
}

template <class S>
void Tree<S>::resize(size_t _num) {
    numnodes = _num;
    x.resize(numnodes);
    y.resize(numnodes);
    s.resize(numnodes);
    r.resize(numnodes);
    m.resize(numnodes);
    ioffset.resize(numnodes);
    num.resize(numnodes);
    std::fill(num.begin(), num.end(), 0);
    epoffset.resize(numnodes);
    epnum.resize(numnodes);
}

template <class S>
void Tree<S>::print(size_t _num) {
    printf("\nTree with %d levels\n",levels);
    for(size_t i=1; i<numnodes && i<_num; ++i) {
        printf("  %ld  %ld %ld  %g\n",i, num[i], ioffset[i], s[i]);
    }
}


//
// The inner, scalar kernel
//
template <class S, class A>
static inline void nbody_kernel(const S sx, const S sy,
                                const S sr, const S sm,
                                const S tx, const S ty,
                                A& __restrict__ tax, A& __restrict__ tay) {
    // 12 flops
    const S dx = tx - sx;
    const S dy = ty - sy;
    S r2 = dx*dx + dy*dy + sr*sr;
    r2 = sm/r2;
    tax -= r2 * dy;
    tay += r2 * dx;
}

//
// A blocked kernel
//
//static inline void nbody_src_block(const float sx, const float sy, const float sz,

//
// Caller for the O(N^2) kernel
//
template <class S, class A>
void nbody_naive(const Parts<S,A>& __restrict__ srcs, Parts<S,A>& __restrict__ targs, const size_t tskip) {
    #pragma omp parallel for
    for (size_t i = 0; i < targs.n; i+=tskip) {
        targs.u[i] = 0.0;
        targs.v[i] = 0.0;
        //#pragma clang loop vectorize(enable) interleave(enable)
        for (size_t j = 0; j < srcs.n; j++) {
            nbody_kernel(srcs.x[j], srcs.y[j], srcs.r[j], srcs.m[j],
                         targs.x[i], targs.y[i],
                         targs.u[i], targs.v[i]);
        }
    }
}

//
// Recursive kernel for the treecode using 1st order box approximations
//
template <class S, class A>
void treecode1_block(const Parts<S,A>& sp, const Tree<S>& st, const size_t tnode, const float theta,
                     const S tx, const S ty,
                     A& tax, A& tay) {

    // if box is a leaf node, just compute the influence and return
    if (st.num[tnode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        for (size_t j = st.ioffset[tnode]; j < st.ioffset[tnode] + st.num[tnode]; j++) {
            nbody_kernel(sp.x[j], sp.y[j], sp.r[j], sp.m[j],
                         tx, ty, tax, tay);
        }
        return;
    }

    // distance from box center of mass to target point
    const S dx = st.x[tnode] - tx;
    const S dy = st.y[tnode] - ty;
    const S dist = std::sqrt(dx*dx + dy*dy);

    // is source tree node far enough away?
    if (dist / st.s[tnode] > theta) {
        // box is far enough removed, approximate its influence
        nbody_kernel(st.x[tnode], st.y[tnode], 0.0f, st.m[tnode],
                     tx, ty, tax, tay);
    } else {
        // box is too close, open up its children
        (void) treecode1_block(sp, st, 2*tnode,   theta, tx, ty, tax, tay);
        (void) treecode1_block(sp, st, 2*tnode+1, theta, tx, ty, tax, tay);
    }
}

//
// Caller for the simple O(NlogN) kernel
//
template <class S, class A>
void nbody_treecode1(const Parts<S,A>& srcs, const Tree<S>& stree, Parts<S,A>& targs, const float theta) {
    #pragma omp parallel for
    for (size_t i = 0; i < targs.n; i++) {
        targs.u[i] = 0.0;
        targs.v[i] = 0.0;
        treecode1_block(srcs, stree, 1, theta,
                        targs.x[i], targs.y[i],
                        targs.u[i], targs.v[i]);
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
template <class S, class A>
void treecode2_block(const Parts<S,A>& sp, const Parts<S,A>& ep,
                     const Tree<S>& st, const size_t tnode, const float theta,
                     const S tx, const S ty,
                     A& tax, A& tay,
                     struct treecode2_stats& stats) {

    // if box is a leaf node, just compute the influence and return
    if (st.num[tnode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        for (size_t j = st.ioffset[tnode]; j < st.ioffset[tnode] + st.num[tnode]; j++) {
            nbody_kernel(sp.x[j], sp.y[j], sp.r[j], sp.m[j],
                         tx, ty, tax, tay);
        }
        stats.sltp++;
        return;
    }

    // distance from box center of mass to target point
    const S dx = st.x[tnode] - tx;
    const S dy = st.y[tnode] - ty;
    const S dist = std::sqrt(dx*dx + dy*dy);

    // is source tree node far enough away?
    if (dist / st.s[tnode] > theta) {
        // this version uses equivalent points instead!
        for (size_t j = st.epoffset[tnode]; j < st.epoffset[tnode] + st.epnum[tnode]; j++) {
            nbody_kernel(ep.x[j], ep.y[j], ep.r[j], ep.m[j],
                         tx, ty, tax, tay);
        }
        stats.sbtp++;
    } else {
        // box is too close, open up its children
        (void) treecode2_block(sp, ep, st, 2*tnode,   theta, tx, ty, tax, tay, stats);
        (void) treecode2_block(sp, ep, st, 2*tnode+1, theta, tx, ty, tax, tay, stats);
    }
}

//
// Caller for the better (equivalent particle) O(NlogN) kernel
//
template <class S, class A>
float nbody_treecode2(const Parts<S,A>& srcs, const Parts<S,A>& eqsrcs,
                      const Tree<S>& stree, Parts<S,A>& targs, const float theta) {

    struct treecode2_stats stats = {0, 0};

    #pragma omp parallel
    {
        struct treecode2_stats threadstats = {0, 0};

        #pragma omp for
        for (size_t i = 0; i < targs.n; i++) {
            targs.u[i] = 0.0;
            targs.v[i] = 0.0;
            treecode2_block(srcs, eqsrcs, stree, 1, theta,
                            targs.x[i], targs.y[i],
                            targs.u[i], targs.v[i],
                            threadstats);
        }

        #pragma omp critical
        {
            stats.sltp += threadstats.sltp;
            stats.sbtp += threadstats.sbtp;
        }
    }

    printf("  %ld target particles averaged %g leaf-part and %g equiv-part interactions\n",
           targs.n, stats.sltp/(float)targs.n, stats.sbtp/(float)targs.n);
    //printf("  sltp %ld  sbtp %ld\n", stats.sltp, stats.sbtp);

    return 12.f * ((float)stats.sltp + (float)stats.sbtp) * (float)blockSize;
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
template <class S, class A>
void splitNode(Parts<S,A>& p, size_t pfirst, size_t plast, Tree<S>& t, size_t tnode) {

    //printf("\nsplitNode %d  %ld %ld\n", tnode, pfirst, plast);
    //printf("splitNode %d  %ld %ld\n", tnode, pfirst, plast);
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
    //if (pfirst == 0) reset_and_start_timer();
    std::vector<S> boxsizes(2);
    auto minmax = minMaxValue(p.x, pfirst, plast);
    boxsizes[0] = minmax.second - minmax.first;
    //printf("  node x min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.y, pfirst, plast);
    boxsizes[1] = minmax.second - minmax.first;
    //printf("       y min/max %g %g\n", minmax.first, minmax.second);

    // find total mass and center of mass
    //printf("find mass/cm\n");
    //if (pfirst == 0) reset_and_start_timer();
    t.m[tnode] = std::accumulate(p.m.begin()+pfirst, p.m.begin()+plast, 0.0);

    // copy strength vector
    alignas(32) decltype(p.m) absstr(p.m.begin()+pfirst, p.m.begin()+plast);
    // find abs() of each entry using a lambda
    std::for_each(absstr.begin(), absstr.end(), [](float &str){ str = std::abs(str); });

    // sum of abs of strengths
    auto nodestr = std::accumulate(absstr.begin(), absstr.end(), 0.0);

    t.x[tnode] = std::inner_product(p.x.begin()+pfirst, p.x.begin()+plast, absstr.begin(), 0.0) / nodestr;
    t.y[tnode] = std::inner_product(p.y.begin()+pfirst, p.y.begin()+plast, absstr.begin(), 0.0) / nodestr;
    //printf("  total mass %g abs mass %g and cm %g %g\n", t.m[tnode], nodestr, t.x[tnode], t.y[tnode]);

    // write all this data to the tree node
    t.ioffset[tnode] = pfirst;
    t.num[tnode] = plast - pfirst;
    //printf("  tree node has offset %d and num %d\n", t.ioffset[tnode], t.num[tnode]);

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();
    //printf("  longest axis is %ld, length %g\n", maxaxis, boxsizes[maxaxis]);
    t.s[tnode] = 0.5 * std::sqrt(std::pow(boxsizes[0],2) + std::pow(boxsizes[1],2));
    //printf("  tree node time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // no need to split or compute further
    if (t.num[tnode] <= blockSize) {
        // we are at block size!
        //printf("  tree node %ld position %g %g size %g %g\n", tnode, t.x[tnode], t.y[tnode], boxsizes[0], boxsizes[1]);
        return;
    }

    // sort this portion of the array along the big axis
    //printf("sort\n");
    //if (pfirst == 0) reset_and_start_timer();
    if (maxaxis == 0) {
        (void) sortIndexesSection(sort_recursion, p.x, p.itemp, pfirst, plast);
        //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.x[p.itemp[i]]);
    } else if (maxaxis == 1) {
        (void) sortIndexesSection(sort_recursion, p.y, p.itemp, pfirst, plast);
        //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.y[p.itemp[i]]);
    }
    //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, idx[i], p.x[idx[i]]);
    //if (pfirst == 0) printf("    sort time:\t\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // rearrange the elements - parallel sections did not make things faster
    //printf("reorder\n");
    reorder(p.x, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.y, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.m, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.r, p.ftemp, p.itemp, pfirst, plast);
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

    if (tnode == 1) {
        // this is executed on the final call
    }
}

//
// Recursively refine leaf node's particles until they are hierarchically nearby
// Code is borrowed from splitNode above
//
template <class S, class A>
void refineLeaf(Parts<S,A>& p, Tree<S>& t, size_t pfirst, size_t plast) {

    // if there are 1 or 2 particles, then they are already in "order"
    if (plast-pfirst < 3) return;

    // perform very much the same action as tree-build
    //printf("    refining particles %ld to %ld\n", pfirst, plast);

    // find the min/max of the three axes
    std::vector<S> boxsizes(2);
    auto minmax = minMaxValue(p.x, pfirst, plast);
    boxsizes[0] = minmax.second - minmax.first;
    minmax = minMaxValue(p.y, pfirst, plast);
    boxsizes[1] = minmax.second - minmax.first;

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();

    // sort this portion of the array along the big axis
    if (maxaxis == 0) {
        (void) sortIndexesSection(0, p.x, p.itemp, pfirst, plast);
    } else if (maxaxis == 1) {
        (void) sortIndexesSection(0, p.y, p.itemp, pfirst, plast);
    }

    // rearrange the elements
    reorder(p.x, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.y, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.m, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.r, p.ftemp, p.itemp, pfirst, plast);

    // determine where the split should be
    size_t pmiddle = pfirst + (1 << log_2(plast-pfirst-1));

    // recursively call this routine for this node's new children
    (void) refineLeaf(p, t, pfirst,  pmiddle);
    (void) refineLeaf(p, t, pmiddle, plast);
}

//
// Loop over all leaf nodes in the tree and call the refine function on them
//
template <class S, class A>
void refineTree(Parts<S,A>& p, Tree<S>& t, size_t tnode) {
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
template <class S, class A>
void calcEquivalents(Parts<S,A>& p, Parts<S,A>& ep, Tree<S>& t, size_t tnode) {
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
                S str1 = std::abs(ep.m[ip-1]);
                S str2 = std::abs(ep.m[ip]);
                S pairm = str1 + str2;
                ep.x[iep] = (ep.x[ip-1]*str1 + ep.x[ip]*str2) / pairm;
                ep.y[iep] = (ep.y[ip-1]*str1 + ep.y[ip]*str2) / pairm;
                ep.r[iep] = (ep.r[ip-1]*str1 + ep.r[ip]*str2) / pairm;
                ep.m[iep] = ep.m[ip-1] + ep.m[ip];
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.epoffset[ichild]+t.epnum[ichild]) {
                //printf("    passing %d up into %d\n", ip-1,iep);
                ep.x[iep] = ep.x[ip-1];
                ep.y[iep] = ep.y[ip-1];
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
                S str1 = std::abs(p.m[ip-1]);
                S str2 = std::abs(p.m[ip]);
                S pairm = str1 + str2;
                ep.x[iep] = (p.x[ip-1]*str1 + p.x[ip]*str2) / pairm;
                ep.y[iep] = (p.y[ip-1]*str1 + p.y[ip]*str2) / pairm;
                ep.r[iep] = (p.r[ip-1]*str1 + p.r[ip]*str2) / pairm;
                ep.m[iep] = p.m[ip-1] + p.m[ip];
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.ioffset[ichild]+t.num[ichild]) {
                //printf("    passing %d up into %d\n", ip-1,iep);
                ep.x[iep] = p.x[ip-1];
                ep.y[iep] = p.y[ip-1];
                ep.r[iep] = p.r[ip-1];
                ep.m[iep] = p.m[ip-1];
            }
            t.epnum[tnode] += numEqps;
        }
    }

    //printf("  node %d finally has %d equivalent particles, offset %d\n", tnode, t.epnum[tnode], t.epoffset[tnode]);
}

