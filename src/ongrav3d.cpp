/*
 * ongrav3d - testbed for an O(N) 3d gravitational solver
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
    void random_in_cube();

    size_t n;
    // state
    alignas(32) std::vector<S> x;
    alignas(32) std::vector<S> y;
    alignas(32) std::vector<S> z;
    alignas(32) std::vector<S> r;
    // actuator (needed by sources)
    alignas(32) std::vector<S> m;
    // results (needed by targets)
    alignas(32) std::vector<A> u;
    alignas(32) std::vector<A> v;
    alignas(32) std::vector<A> w;
    // temporary
    alignas(32) std::vector<size_t> itemp;
    alignas(32) std::vector<S> ftemp;
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
    z.resize(n);
    r.resize(n);
    m.resize(n);
    u.resize(n);
    v.resize(n);
    w.resize(n);
    itemp.resize(n);
    ftemp.resize(n);
}

template <class S, class A>
void Parts<S,A>::random_in_cube() {
    for (auto&& _x : x) { _x = (S)rand()/(S)RAND_MAX; }
    for (auto&& _y : y) { _y = (S)rand()/(S)RAND_MAX; }
    for (auto&& _z : z) { _z = (S)rand()/(S)RAND_MAX; }
    for (auto&& _r : r) { _r = 1.0f / cbrt((S)n); }
    for (auto&& _m : m) { _m = 2.0f*(S)rand()/(S)RAND_MAX / (S)n; }
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

    // number of levels in the tree
    int levels;
    // number of nodes in the tree (always 2^l)
    int numnodes;

    // tree node centers (of mass?)
    alignas(32) std::vector<S> x;
    alignas(32) std::vector<S> y;
    alignas(32) std::vector<S> z;
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
    z.resize(numnodes);
    s.resize(numnodes);
    r.resize(numnodes);
    m.resize(numnodes);
    ioffset.resize(numnodes);
    num.resize(numnodes);
    std::fill(num.begin(), num.end(), 0);
    epoffset.resize(numnodes);
    epnum.resize(numnodes);
}



//
// The inner, scalar kernel
//
template <class S, class A>
static inline void nbody_kernel(const S sx, const S sy, const S sz,
                                const S sr, const S sm,
                                const S tx, const S ty, const S tz,
                                A& __restrict__ tax, A& __restrict__ tay, A& __restrict__ taz) {
    // 19 flops
    const S dx = sx - tx;
    const S dy = sy - ty;
    const S dz = sz - tz;
    S r2 = dx*dx + dy*dy + dz*dz + sr*sr;
    r2 = sm/(r2*sqrt(r2));
    tax += r2 * dx;
    tay += r2 * dy;
    taz += r2 * dz;
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
        targs.w[i] = 0.0;
        //#pragma clang loop vectorize(enable) interleave(enable)
        for (size_t j = 0; j < srcs.n; j++) {
            nbody_kernel(srcs.x[j], srcs.y[j], srcs.z[j], srcs.r[j], srcs.m[j],
                         targs.x[i], targs.y[i], targs.z[i],
                         targs.u[i], targs.v[i], targs.w[i]);
        }
    }
}

//
// Recursive kernel for the treecode using 1st order box approximations
//
template <class S, class A>
void treecode1_block(const Parts<S,A>& sp, const Tree<S>& st, const size_t tnode, const float theta,
                     const S tx, const S ty, const S tz,
                     A& tax, A& tay, A& taz) {

    // if box is a leaf node, just compute the influence and return
    if (st.num[tnode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        for (size_t j = st.ioffset[tnode]; j < st.ioffset[tnode] + st.num[tnode]; j++) {
            nbody_kernel(sp.x[j], sp.y[j], sp.z[j], sp.r[j], sp.m[j],
                         tx, ty, tz, tax, tay, taz);
        }
        return;
    }

    // distance from box center of mass to target point
    const S dx = st.x[tnode] - tx;
    const S dy = st.y[tnode] - ty;
    const S dz = st.z[tnode] - tz;
    const S dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // is source tree node far enough away?
    if (dist / st.s[tnode] > theta) {
        // box is far enough removed, approximate its influence
        nbody_kernel(st.x[tnode], st.y[tnode], st.z[tnode], 0.0f, st.m[tnode],
                     tx, ty, tz, tax, tay, taz);
    } else {
        // box is too close, open up its children
        (void) treecode1_block(sp, st, 2*tnode,   theta, tx, ty, tz, tax, tay, taz);
        (void) treecode1_block(sp, st, 2*tnode+1, theta, tx, ty, tz, tax, tay, taz);
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
        targs.w[i] = 0.0;
        treecode1_block(srcs, stree, 1, theta,
                        targs.x[i], targs.y[i], targs.z[i],
                        targs.u[i], targs.v[i], targs.w[i]);
    }
}

//
// Recursive kernel for the treecode using equivalent particles
//
template <class S, class A>
void treecode2_block(const Parts<S,A>& sp, const Parts<S,A>& ep,
                     const Tree<S>& st, const size_t tnode, const float theta,
                     const S tx, const S ty, const S tz,
                     A& tax, A& tay, A& taz) {

    static int sltp = 0;
    static int sbtp = 0;

    // report on interactions
    if (tnode == 0) {
        int tlc = sp.n;
        printf("%d target particles averaged %g leaf-part and %g equiv-part interactions\n",
               tlc, sltp/(float)tlc, sbtp/(float)tlc);
        printf("  sltp %d  sbtp %d\n", sltp, sbtp);

        return;
    }

    // if box is a leaf node, just compute the influence and return
    if (st.num[tnode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        for (size_t j = st.ioffset[tnode]; j < st.ioffset[tnode] + st.num[tnode]; j++) {
            nbody_kernel(sp.x[j], sp.y[j], sp.z[j], sp.r[j], sp.m[j],
                         tx, ty, tz, tax, tay, taz);
        }
        sltp++;
        return;
    }

    // distance from box center of mass to target point
    const S dx = st.x[tnode] - tx;
    const S dy = st.y[tnode] - ty;
    const S dz = st.z[tnode] - tz;
    const S dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // is source tree node far enough away?
    if (dist / st.s[tnode] > theta) {
        // this version uses equivalent points instead!
        for (size_t j = st.epoffset[tnode]; j < st.epoffset[tnode] + st.epnum[tnode]; j++) {
            nbody_kernel(ep.x[j], ep.y[j], ep.z[j], ep.r[j], ep.m[j],
                         tx, ty, tz, tax, tay, taz);
        }
        sbtp++;
    } else {
        // box is too close, open up its children
        (void) treecode2_block(sp, ep, st, 2*tnode,   theta, tx, ty, tz, tax, tay, taz);
        (void) treecode2_block(sp, ep, st, 2*tnode+1, theta, tx, ty, tz, tax, tay, taz);
    }
}

//
// Caller for the better (equivalent particle) O(NlogN) kernel
//
template <class S, class A>
void nbody_treecode2(const Parts<S,A>& srcs, const Parts<S,A>& eqsrcs,
                     const Tree<S>& stree, Parts<S,A>& targs, const float theta) {
    #pragma omp parallel for
    for (size_t i = 0; i < targs.n; i++) {
        targs.u[i] = 0.0;
        targs.v[i] = 0.0;
        targs.w[i] = 0.0;
        treecode2_block(srcs, eqsrcs, stree, 1, theta,
                        targs.x[i], targs.y[i], targs.z[i],
                        targs.u[i], targs.v[i], targs.w[i]);
    }

    // report on the number of interactions
    treecode2_block(srcs, eqsrcs, stree, 0, theta,
                    targs.x[0], targs.y[0], targs.z[0],
                    targs.u[0], targs.v[0], targs.w[0]);
}

//
// Approximate a spatial derivative from a number of irregularly-spaced points
//
template <class S, class A>
A least_squares_val(const S xt, const S yt, const S zt,
                    const std::vector<S>& x, const std::vector<S>& y,
                    const std::vector<S>& z, const std::vector<A>& u,
                    const size_t istart, const size_t iend) {

    //printf("  target point at %g %g %g\n", xt, yt, zt);
    S sn = 0.0f;
    S sx = 0.0f;
    S sy = 0.0f;
    S sz = 0.0f;
    S sx2 = 0.0f;
    S sy2 = 0.0f;
    S sz2 = 0.0f;
    S sv = 0.0f;
    S sxv = 0.0f;
    S syv = 0.0f;
    S szv = 0.0f;
    S sxy = 0.0f;
    S sxz = 0.0f;
    S syz = 0.0f;
    for (size_t i=istart; i<iend; ++i) {
        const S dx = x[i] - xt;
        const S dy = y[i] - yt;
        const S dz = z[i] - zt;
        const S dist = sqrt(dx*dx+dy*dy+dz*dz);
        //printf("    point %d at %g %g %g dist %g with value %g\n", i, x[i], y[i], z[i], u[i]);
        //printf("    point %d at %g %g %g dist %g with value %g\n", i, dx, dy, dz, dist, u[i]);
        const S weight = 1.f / (0.001f + dist);
        //const float oods = 1.0f / 
        //nsum
        // see https://en.wikipedia.org/wiki/Linear_least_squares_%28mathematics%29
        // must solve a system of equations for ax + by + cz + d = 0
        // while minimizing the square error, this is a 4x4 matrix solve
        // ideally while also weighting the data points by their distance

        // compute sums of moments
        //const float weight = 1.f;
        sn += weight;
        sx += weight*dx;
        sy += weight*dy;
        sz += weight*dz;
        sv += weight*u[i];
        sxy += weight*dx*dy;
        sxz += weight*dx*dz;
        syz += weight*dy*dz;
        sxv += weight*dx*u[i];
        syv += weight*dy*u[i];
        szv += weight*dz*u[i];
        sx2 += weight*dx*dx;
        sy2 += weight*dy*dy;
        sz2 += weight*dz*dz;
    }
    // 47 flops per iter
    //printf("    sums are %g %g %g %g %g ...\n", sx, sy, sz, sv, sxy);

    // now begin to solve the equation
    const S i1 = sx/sxz - sn/sz;
    const S i2 = sx2/sxz - sx/sz;
    const S i3 = sxy/sxz - sy/sz;
    const S i4 = sxv/sxz - sv/sz;
    const S j1 = sy/syz - sn/sz;
    const S j2 = sxy/syz - sx/sz;
    const S j3 = sy2/syz - sy/sz;
    const S j4 = syv/syz - sv/sz;
    const S k1 = sz/sz2 - sn/sz;
    const S k2 = sxz/sz2 - sx/sz;
    const S k3 = syz/sz2 - sy/sz;
    const S k4 = szv/sz2 - sv/sz;
    const S q1 = i3*j1 - i1*j3;
    const S q2 = i3*j2 - i2*j3;
    const S q3 = i3*j4 - i4*j3;
    const S r1 = i3*k1 - i1*k3;
    const S r2 = i3*k2 - i2*k3;
    const S r3 = i3*k4 - i4*k3;
    // 18*3 = 54 flops

    const A b1 = (r2*q3 - r3*q2) / (r2*q1 - r1*q2);
    // 7 more
    //printf("    b1 is %g\n", b1);
    //const float b2 = r3/r2 - b1*r1/r2;
    //printf("    b2 is %g\n", b2);
    //const float b3 = j4/j3 - b1*j1/j3 - b2*j2/j3;
    //printf("    b3 is %g\n", b3);
    //const float b4 = sv/sz - b1/sz - b2*sx/sz - b3*sy/sz;
    //printf("    b4 is %g\n", b4);

    // when 16 contributing points, this is 813 flops

    //if (fabs(u[istart]) > 0.0) exit(0);
    return b1;
}

//
// Caller for the fast summation O(N) method
//
// ittn is the target tree node that this routine will work on
// itsv is the source tree node vector that will affect ittn
//
// We will change u,v,w for the targs points and the eqtargs equivalent points
//
template <class S, class A>
void nbody_fastsumm(const Parts<S,A>& srcs, const Parts<S,A>& eqsrcs, const Tree<S>& stree,
                    Parts<S,A>& targs, Parts<S,A>& eqtargs, const Tree<S>& ttree,
                    const size_t ittn, std::vector<size_t> istv_in, const float theta) {

    static int sltl = 0;
    static int sbtl = 0;
    static int sltb = 0;
    static int sbtb = 0;
    static int tlc = 0;
    static int lpc = 0;
    static int bpc = 0;

    // start counters
    if (ittn == 1) {
        sltl = 0;
        sbtl = 0;
        sltb = 0;
        sbtb = 0;
        tlc = 0;
        lpc = 0;
        bpc = 0;
    }

    // quit out if we've gone too far
    if (ttree.num[ittn] < 1) return;

    //printf("Targ box %d is affected by %lu source boxes at this level\n",ittn,istv.size());
    const bool targetIsLeaf = ttree.num[ittn] <= blockSize;

    // prepare the target arrays for accumulations
    if (targetIsLeaf) {
        tlc++;
        // zero the velocities
        std::fill_n(&(targs.u[ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);
        std::fill_n(&(targs.v[ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);
        std::fill_n(&(targs.w[ttree.ioffset[ittn]]), ttree.num[ittn], 0.0f);

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
                    targs.u[idest] = least_squares_val(targs.x[idest], targs.y[idest], targs.z[idest],
                                                       eqtargs.x, eqtargs.y, eqtargs.z, eqtargs.u, istart, iend);
                    targs.v[idest] = least_squares_val(targs.x[idest], targs.y[idest], targs.z[idest],
                                                       eqtargs.x, eqtargs.y, eqtargs.z, eqtargs.v, istart, iend);
                    targs.w[idest] = least_squares_val(targs.x[idest], targs.y[idest], targs.z[idest],
                                                       eqtargs.x, eqtargs.y, eqtargs.z, eqtargs.w, istart, iend);
                } else {
                    // as a first take, simply copy the result to the children
                    targs.u[idest] = eqtargs.u[iorig];
                    targs.v[idest] = eqtargs.v[iorig];
                    targs.w[idest] = eqtargs.w[iorig];
                }
            }
            lpc++;
        }

    } else {
        // zero the equivalent particle velocities
        std::fill_n(&(eqtargs.u[ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);
        std::fill_n(&(eqtargs.v[ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);
        std::fill_n(&(eqtargs.w[ttree.epoffset[ittn]]), ttree.epnum[ittn], 0.0f);

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
                    eqtargs.u[idest] = least_squares_val(eqtargs.x[idest], eqtargs.y[idest], eqtargs.z[idest],
                                                         eqtargs.x, eqtargs.y, eqtargs.z, eqtargs.u, istart, iend);
                    eqtargs.v[idest] = least_squares_val(eqtargs.x[idest], eqtargs.y[idest], eqtargs.z[idest],
                                                         eqtargs.x, eqtargs.y, eqtargs.z, eqtargs.v, istart, iend);
                    eqtargs.w[idest] = least_squares_val(eqtargs.x[idest], eqtargs.y[idest], eqtargs.z[idest],
                                                         eqtargs.x, eqtargs.y, eqtargs.z, eqtargs.w, istart, iend);
                } else {
                    // as a first take, simply copy the result to the children
                    eqtargs.u[idest] = eqtargs.u[iorig];
                    eqtargs.v[idest] = eqtargs.v[iorig];
                    eqtargs.w[idest] = eqtargs.w[iorig];
                }
            }
            bpc++;
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
                nbody_kernel(srcs.x[j],  srcs.y[j],  srcs.z[j], srcs.r[j], srcs.m[j],
                             targs.x[i], targs.y[i], targs.z[i],
                             targs.u[i], targs.v[i], targs.w[i]);
            }
            }
            sltl++;
            continue;
        }

        // distance from box center of mass to target point
        const S dx = stree.x[sn] - ttree.x[ittn];
        const S dy = stree.y[sn] - ttree.y[ittn];
        const S dz = stree.z[sn] - ttree.z[ittn];
        const S halfsize = 0.5*(stree.s[sn] + ttree.s[ittn]);
        const S dist = sqrtf(dx*dx + dy*dy + dz*dz);
        //printf("  src box %d is %g away and halfsize %g\n",sn, dist, halfsize);

        // split on what to do with this pair
        if ((dist-halfsize) / halfsize > theta) {
            // it is far enough - we can approximate
            //printf("    well-separated\n");

            if (sourceIsLeaf) {
                // compute real source particles on equivalent target points
                for (size_t i = ttree.epoffset[ittn]; i < ttree.epoffset[ittn] + ttree.epnum[ittn]; i++) {
                for (size_t j = stree.ioffset[sn];    j < stree.ioffset[sn]    + stree.num[sn];     j++) {
                    nbody_kernel(srcs.x[j],    srcs.y[j],    srcs.z[j], srcs.r[j], srcs.m[j],
                                 eqtargs.x[i], eqtargs.y[i], eqtargs.z[i],
                                 eqtargs.u[i], eqtargs.v[i], eqtargs.w[i]);
                }
                }
                sltb++;

            } else if (targetIsLeaf) {
                // compute equivalent source particles on real target points
                for (size_t i = ttree.ioffset[ittn]; i < ttree.ioffset[ittn] + ttree.num[ittn]; i++) {
                for (size_t j = stree.epoffset[sn];  j < stree.epoffset[sn]  + stree.epnum[sn]; j++) {
                    nbody_kernel(eqsrcs.x[j], eqsrcs.y[j], eqsrcs.z[j], eqsrcs.r[j], eqsrcs.m[j],
                                 targs.x[i],  targs.y[i],  targs.z[i],
                                 targs.u[i],  targs.v[i],  targs.w[i]);
                }
                }
                sbtl++;

            } else {
                // compute equivalent source particles on equivalent target points
                for (size_t i = ttree.epoffset[ittn]; i < ttree.epoffset[ittn] + ttree.epnum[ittn]; i++) {
                for (size_t j = stree.epoffset[sn];   j < stree.epoffset[sn]   + stree.epnum[sn];   j++) {
                    nbody_kernel(eqsrcs.x[j],  eqsrcs.y[j],  eqsrcs.z[j], eqsrcs.r[j], eqsrcs.m[j],
                                 eqtargs.x[i], eqtargs.y[i], eqtargs.z[i],
                                 eqtargs.u[i], eqtargs.v[i], eqtargs.w[i]);
                }
                }
                sbtb++;
            }

        } else if (ttree.s[ittn] > stree.s[sn]) {
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

    if (not targetIsLeaf) {
        // prolongation of equivalent particle velocities to children's equivalent particles

        // recurse onto the target box's children
        // can't use reduction(+:sltl,sbtl,sltb,sbtb,tlc,lpc,bpc) for tasks
        #pragma omp task shared(srcs,eqsrcs,stree,targs,eqtargs,ttree)
        (void) nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree, 2*ittn, cstv, theta);
        #pragma omp task shared(srcs,eqsrcs,stree,targs,eqtargs,ttree)
        (void) nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree, 2*ittn+1, cstv, theta);
        //#pragma omp taskwait
    }

    // report counter results
    if (ittn == 1) {
        #pragma omp taskwait
        printf("%d target leaf nodes averaged %g leaf-leaf and %g equiv-leaf interactions\n",
               tlc, sltl/(float)tlc, sbtl/(float)tlc);
        printf("  sltl %d  sbtl %d  sltb %d  sbtb %d\n", sltl, sbtl, sltb, sbtb);
        printf("  leaf prolongation count %d  box pc %d\n", lpc, bpc);
    }

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

  //if (istart == 0) printf("    inside sortIndexesSection with level %d\n", recursion_level);

  // initialize original index locations
  std::iota(idx.begin()+istart, idx.begin()+istop, istart);

  // sort indexes based on comparing values in v, possibly with forking
  splitSort(recursion_level, v, idx, istart, istop);
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
    std::vector<S> boxsizes(3);
    auto minmax = minMaxValue(p.x, pfirst, plast);
    boxsizes[0] = minmax.second - minmax.first;
    //printf("  node x min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.y, pfirst, plast);
    boxsizes[1] = minmax.second - minmax.first;
    //printf("       y min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.z, pfirst, plast);
    boxsizes[2] = minmax.second - minmax.first;
    //printf("       z min/max %g %g\n", minmax.first, minmax.second);
    //if (pfirst == 0) printf("    minmax time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // find total mass and center of mass
    //printf("find mass/cm\n");
    //if (pfirst == 0) reset_and_start_timer();
    t.m[tnode] = std::accumulate(p.m.begin()+pfirst, p.m.begin()+plast, 0.0);
    t.x[tnode] = std::inner_product(p.x.begin()+pfirst, p.x.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    t.y[tnode] = std::inner_product(p.y.begin()+pfirst, p.y.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    t.z[tnode] = std::inner_product(p.z.begin()+pfirst, p.z.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    //printf("  total mass %g and cm %g %g %g\n", t.m[tnode], t.x[tnode], t.y[tnode], t.z[tnode]);
    //if (pfirst == 0) printf("    inner product time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // write all this data to the tree node
    t.ioffset[tnode] = pfirst;
    t.num[tnode] = plast - pfirst;
    //printf("  tree node has offset %d and num %d\n", t.ioffset[tnode], t.num[tnode]);

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();
    //printf("  longest axis is %ld, length %g\n", maxaxis, boxsizes[maxaxis]);
    t.s[tnode] = boxsizes[maxaxis];
    //printf("  tree node time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // no need to split or compute further
    if (t.num[tnode] <= blockSize) return;

    // sort this portion of the array along the big axis
    //printf("sort\n");
    //if (pfirst == 0) reset_and_start_timer();
    if (maxaxis == 0) {
        (void) sortIndexesSection(sort_recursion, p.x, p.itemp, pfirst, plast);
        //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.x[p.itemp[i]]);
    } else if (maxaxis == 1) {
        (void) sortIndexesSection(sort_recursion, p.y, p.itemp, pfirst, plast);
        //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.y[p.itemp[i]]);
    } else if (maxaxis == 2) {
        (void) sortIndexesSection(sort_recursion, p.z, p.itemp, pfirst, plast);
        //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.z[p.itemp[i]]);
    }
    //for (size_t i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, idx[i], p.x[idx[i]]);
    //if (pfirst == 0) printf("    sort time:\t\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // rearrange the elements - parallel sections did not make things faster
    //printf("reorder\n");
    reorder(p.x, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.y, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.z, p.ftemp, p.itemp, pfirst, plast);
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
    std::vector<S> boxsizes(3);
    auto minmax = minMaxValue(p.x, pfirst, plast);
    boxsizes[0] = minmax.second - minmax.first;
    minmax = minMaxValue(p.y, pfirst, plast);
    boxsizes[1] = minmax.second - minmax.first;
    minmax = minMaxValue(p.z, pfirst, plast);
    boxsizes[2] = minmax.second - minmax.first;

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();

    // sort this portion of the array along the big axis
    if (maxaxis == 0) {
        (void) sortIndexesSection(0, p.x, p.itemp, pfirst, plast);
    } else if (maxaxis == 1) {
        (void) sortIndexesSection(0, p.y, p.itemp, pfirst, plast);
    } else if (maxaxis == 2) {
        (void) sortIndexesSection(0, p.z, p.itemp, pfirst, plast);
    }

    // rearrange the elements
    reorder(p.x, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.y, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.z, p.ftemp, p.itemp, pfirst, plast);
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
                S pairm = ep.m[ip-1] + ep.m[ip];
                ep.x[iep] = (ep.x[ip-1]*ep.m[ip-1] + ep.x[ip]*ep.m[ip]) / pairm;
                ep.y[iep] = (ep.y[ip-1]*ep.m[ip-1] + ep.y[ip]*ep.m[ip]) / pairm;
                ep.z[iep] = (ep.z[ip-1]*ep.m[ip-1] + ep.z[ip]*ep.m[ip]) / pairm;
                ep.r[iep] = (ep.r[ip-1]*ep.m[ip-1] + ep.r[ip]*ep.m[ip]) / pairm;
                ep.m[iep] = pairm;
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.epoffset[ichild]+t.epnum[ichild]) {
                //printf("    passing %d up into %d\n", ip-1,iep);
                ep.x[iep] = ep.x[ip-1];
                ep.y[iep] = ep.y[ip-1];
                ep.z[iep] = ep.z[ip-1];
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
                S pairm = p.m[ip-1] + p.m[ip];
                ep.x[iep] = (p.x[ip-1]*p.m[ip-1] + p.x[ip]*p.m[ip]) / pairm;
                ep.y[iep] = (p.y[ip-1]*p.m[ip-1] + p.y[ip]*p.m[ip]) / pairm;
                ep.z[iep] = (p.z[ip-1]*p.m[ip-1] + p.z[ip]*p.m[ip]) / pairm;
                ep.r[iep] = (p.r[ip-1]*p.m[ip-1] + p.r[ip]*p.m[ip]) / pairm;
                ep.m[iep] = pairm;
            }
            // don't merge the last odd one, just pass it up unmodified
            if (ip == t.ioffset[ichild]+t.num[ichild]) {
                //printf("    passing %d up into %d\n", ip-1,iep);
                ep.x[iep] = p.x[ip-1];
                ep.y[iep] = p.y[ip-1];
                ep.z[iep] = p.z[ip-1];
                ep.r[iep] = p.r[ip-1];
                ep.m[iep] = p.m[ip-1];
            }
            t.epnum[tnode] += numEqps;
        }
    }

    //printf("  node %d finally has %d equivalent particles, offset %d\n", tnode, t.epnum[tnode], t.epoffset[tnode]);
}

//
// basic usage
//
static void usage() {
    fprintf(stderr, "Usage: ongrav3d [-n=<nparticles>]\n");
    exit(1);
}

//
// main routine - run the program
//
int main(int argc, char *argv[]) {

    static std::vector<int> test_iterations = {1, 0, 1, 1};
    bool just_build_trees = true;
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

    // if problem is too big, skip some number of target particles
    size_t ntskip = std::max(1, (int)((float)numSrcs*(float)numTargs/2.e+9));

    printf("Allocate and initialize\n");
    auto start = std::chrono::system_clock::now();

    // allocate space for sources and targets
    Parts<float,double> srcs(numSrcs);
    // initialize particle data
    srcs.random_in_cube();

    Parts<float,double> targs(numTargs);
    // initialize particle data
    targs.random_in_cube();
    for (auto&& m : targs.m) { m = 1.0f; }
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


    // don't need the target tree for treecode, but will for fast code
    printf("\nBuilding the target tree\n");
    start = std::chrono::system_clock::now();
    Tree<float> ttree(numTargs);
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

    // find equivalent points
    printf("\nCalculating equivalent targ points\n");
    start = std::chrono::system_clock::now();
    Parts<float,double> eqtargs((ttree.numnodes/2) * blockSize);
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
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);
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
        nbody_treecode1(srcs, stree, targs, 2.9f);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minTreecode = std::min(minTreecode, dt);
    }
    printf("[onbody treecode]:\t\t[%.4f] seconds\n", minTreecode);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);
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
    printf("RMS error in treecode is %g\n", sqrtf(errsum/errcnt));
    }


    //
    // Run a better O(NlogN) treecode - boxes use equivalent particles
    //
    if (test_iterations[2] > 0) {
    printf("\nRun the treecode O(NlogN) with equivalent particles\n");
    double minTreecode2 = 1e30;
    for (int i = 0; i < test_iterations[2]; ++i) {
        start = std::chrono::system_clock::now();
        nbody_treecode2(srcs, eqsrcs, stree, targs, 1.1f);
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minTreecode2 = std::min(minTreecode2, dt);
    }
    printf("[onbody treecode2]:\t\t[%.4f] seconds\n", minTreecode2);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);
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
    printf("RMS error in treecode2 is %g\n", sqrtf(errsum/errcnt));
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
        #pragma omp parallel
        #pragma omp single
        nbody_fastsumm(srcs, eqsrcs, stree, targs, eqtargs, ttree,
                       1, source_boxes, 2.3f);
        #pragma omp taskwait
        end = std::chrono::system_clock::now(); elapsed_seconds = end-start;
        double dt = elapsed_seconds.count();
        printf("  this run time:\t\t[%.4f] seconds\n", dt);
        minFast = std::min(minFast, dt);
    }
    printf("[onbody fast]:\t\t\t[%.4f] seconds\n", minFast);
    // write sample results
    for (size_t i = 0; i < 4*ntskip; i+=ntskip) printf("   particle %ld vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);
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
    printf("RMS error in fastsumm is %g\n", sqrtf(errsum/errcnt));
    }

    return 0;
}
