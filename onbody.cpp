/*
 * onbody - testbed for an O(N) 3d gravitational solver
 *
 * Copyright (c) 2017, Mark J Stock
 */

#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <algorithm>	// for sort and minmax
#include <numeric>		// for iota
#include "timing.h"

const int blockSize = 64;

//
// A set of particles, can be sources or targets
//
struct Parts {
    int n;
    // state
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> r;
    // actuator
    std::vector<float> m;
    // results
    std::vector<float> u;
    std::vector<float> v;
    std::vector<float> w;
    // temporary
    std::vector<size_t> itemp;
    std::vector<float> ftemp;
};

//
// A tree, made of a structure of arrays
//
// 0 is empty, root node is 1, children are 2,3, their children 4,5 and 6,7
// arrays always have 2^levels boxes allocated, even if some are not used
// this way, node i children are 2*i and 2*i+1
//
struct Tree {
    // number of levels in the tree
    int levels;
    // number of nodes in the tree (always 2^l)
    int numnodes;
    // tree node centers (of mass?)
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    // node size
    std::vector<float> s;
    // node particle radius
    std::vector<float> r;
    // node masses
    std::vector<float> m;

    // real point offset and count
    std::vector<int> ioffset;		// is this redundant?
    std::vector<int> num;
    // equivalent point offset and count
    std::vector<int> epoffset;		// is this redundant?
    std::vector<int> epnum;
};

//
// Find index of msb of uint32
// from http://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
//
static inline uint32_t log_2(const uint32_t x) {
    if (x == 0) return 0;
    return (31 - __builtin_clz (x));
}

//
// The inner, scalar kernel
//
static inline void nbody_kernel(const float sx, const float sy, const float sz,
                                const float sr, const float sm,
                                const float tx, const float ty, const float tz,
                                float& tax, float& tay, float& taz) {
    // 19 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    const float dz = sz - tz;
    float r2 = dx*dx + dy*dy + dz*dz + sr*sr;
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
void nbody_naive(const Parts& srcs, Parts& targs) {
    #pragma omp parallel for
    for (int i = 0; i < targs.n; i++) {
        targs.u[i] = 0.0f;
        targs.v[i] = 0.0f;
        targs.w[i] = 0.0f;
        for (int j = 0; j < srcs.n; j++) {
            nbody_kernel(srcs.x[j], srcs.y[j], srcs.z[j], srcs.r[j], srcs.m[j],
                         targs.x[i], targs.y[i], targs.z[i],
                         targs.u[i], targs.v[i], targs.w[i]);
        }
    }
}

//
// Recursive kernel for the treecode using 1st order box approximations
//
void treecode1_block(const Parts& sp, const Tree& st, const int tnode, const float theta,
                     const float tx, const float ty, const float tz,
                     float& tax, float& tay, float& taz) {

    // distance from box center of mass to target point
    const float dx = st.x[tnode] - tx;
    const float dy = st.y[tnode] - ty;
    const float dz = st.z[tnode] - tz;
    const float dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // is source tree node far enough away?
    if (dist / st.s[tnode] > theta) {
        // box is far enough removed, approximate its influence
        nbody_kernel(st.x[tnode], st.y[tnode], st.z[tnode], 0.0f, st.m[tnode],
                     tx, ty, tz, tax, tay, taz);
    } else if (st.num[tnode] <= blockSize) {
        // box is too close and is a leaf node, look at individual particles
        for (int j = st.ioffset[tnode]; j < st.ioffset[tnode] + st.num[tnode]; j++) {
            nbody_kernel(sp.x[j], sp.y[j], sp.z[j], sp.r[j], sp.m[j],
                         tx, ty, tz, tax, tay, taz);
        }
    } else {
        // box is too close, open up its children
        (void) treecode1_block(sp, st, 2*tnode,   theta, tx, ty, tz, tax, tay, taz);
        (void) treecode1_block(sp, st, 2*tnode+1, theta, tx, ty, tz, tax, tay, taz);
    }
}

//
// Caller for the O(NlogN) kernel
//
void nbody_treecode1(const Parts& srcs, const Tree& stree, Parts& targs, const float theta) {
    #pragma omp parallel for
    for (int i = 0; i < targs.n; i++) {
        targs.u[i] = 0.0f;
        targs.v[i] = 0.0f;
        targs.w[i] = 0.0f;
        treecode1_block(srcs, stree, 1, theta,
                        targs.x[i], targs.y[i], targs.z[i],
                        targs.u[i], targs.v[i], targs.w[i]);
    }
}

//
// Sort but retain only sorted index! Uses C++11 lambdas
// from http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//
std::vector<size_t> sortIndexes(const std::vector<float> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

//
// Sort but retain only sorted index! Uses C++11 lambdas
// from http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//
void sortIndexesSection(const std::vector<float> &v,
                        std::vector<size_t> &idx,
                        const size_t istart, const size_t istop) {

  // initialize original index locations
  std::iota(idx.begin()+istart, idx.begin()+istop, istart);

  // sort indexes based on comparing values in v
  std::sort(idx.begin()+istart, idx.begin()+istop,
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
}

//
// Find min and max values along an axis
//
std::pair<float,float> minMaxValue(const std::vector<float> &x, size_t istart, size_t iend) {

    auto itbeg = x.begin() + istart;
    auto itend = x.begin() + iend;

    auto range = std::minmax_element(itbeg, itend);
    return std::pair<float,float>(x[range.first+istart-itbeg], x[range.second+istart-itbeg]);

    // what's an initializer list?
    //return std::minmax(itbeg, itend);
}

//
// Helper function to reorder a segment of a vector
//
void reorder(std::vector<float> &x, std::vector<float> &t,
             const std::vector<size_t> &idx,
             const size_t pfirst, const size_t plast) {

    // copy the original input float vector x into a temporary vector
    std::copy(x.begin()+pfirst, x.begin()+plast, t.begin()+pfirst);

    // scatter values from the temp vector back into the original vector
    for (int i=pfirst; i<plast; ++i) x[i] = t[idx[i]];
}

//
// Make a VAMsplit k-d tree from this set of particles
// Split this segment of the particles on its longest axis
//
void splitNode(Parts& p, size_t pfirst, size_t plast, Tree& t, int tnode) {

    //printf("\nsplitNode %d  %ld %ld\n", tnode, pfirst, plast);
    //printf("splitNode %d  %ld %ld\n", tnode, pfirst, plast);

    // debug print - starting condition
    //for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);

    // find the min/max of the three axes
    //printf("find min/max\n");
    //reset_and_start_timer();
    std::vector<float> boxsizes(3);
    auto minmax = minMaxValue(p.x, pfirst, plast);
    boxsizes[0] = minmax.second - minmax.first;
    //printf("  node x min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.y, pfirst, plast);
    boxsizes[1] = minmax.second - minmax.first;
    //printf("       y min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.z, pfirst, plast);
    boxsizes[2] = minmax.second - minmax.first;
    //printf("       z min/max %g %g\n", minmax.first, minmax.second);
    //printf("  minmax time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // find total mass and center of mass
    //printf("find mass/cm\n");
    //reset_and_start_timer();
    t.m[tnode] = std::accumulate(p.m.begin()+pfirst, p.m.begin()+plast, 0.0);
    t.x[tnode] = std::inner_product(p.x.begin()+pfirst, p.x.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    t.y[tnode] = std::inner_product(p.y.begin()+pfirst, p.y.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    t.z[tnode] = std::inner_product(p.z.begin()+pfirst, p.z.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    //printf("  total mass %g and cm %g %g %g\n", t.m[tnode], t.x[tnode], t.y[tnode], t.z[tnode]);

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
    //reset_and_start_timer();
    if (maxaxis == 0) {
        (void) sortIndexesSection(p.x, p.itemp, pfirst, plast);
        //for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.x[p.itemp[i]]);
    } else if (maxaxis == 1) {
        (void) sortIndexesSection(p.y, p.itemp, pfirst, plast);
        //for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.y[p.itemp[i]]);
    } else if (maxaxis == 2) {
        (void) sortIndexesSection(p.z, p.itemp, pfirst, plast);
        //for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, p.itemp[i], p.z[p.itemp[i]]);
    }
    //for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, idx[i], p.x[idx[i]]);
    //printf("  sort time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // rearrange the elements
    //printf("reorder\n");
    //reset_and_start_timer();
    reorder(p.x, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.y, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.z, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.m, p.ftemp, p.itemp, pfirst, plast);
    reorder(p.r, p.ftemp, p.itemp, pfirst, plast);
    //for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);
    //printf("  reorder time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // determine where the split should be
    size_t pmiddle = pfirst + blockSize * (1 << log_2((t.num[tnode]-1)/blockSize));
    //printf("split at %ld %ld %ld into nodes %d %d\n", pfirst, pmiddle, plast, 2*tnode, 2*tnode+1);

    // recursively call this routine for this node's new children
    (void) splitNode(p, pfirst,  pmiddle, t, 2*tnode);
    (void) splitNode(p, pmiddle, plast,   t, 2*tnode+1);
}

//
// Recursively refine leaf node's particles until they are hierarchically nearby
// Code is borrowed from splitNode above
//
void refineLeaf(Parts& p, Tree& t, size_t pfirst, size_t plast) {

    // if there are 1 or 2 particles, then they are already in "order"
    if (plast-pfirst < 3) return;

    // perform very much the same action as tree-build
    //printf("    refining particles %ld to %ld\n", pfirst, plast);

    // find the min/max of the three axes
    std::vector<float> boxsizes(3);
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
        (void) sortIndexesSection(p.x, p.itemp, pfirst, plast);
    } else if (maxaxis == 1) {
        (void) sortIndexesSection(p.y, p.itemp, pfirst, plast);
    } else if (maxaxis == 2) {
        (void) sortIndexesSection(p.z, p.itemp, pfirst, plast);
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
void refineTree(Parts& p, Tree& t, int tnode) {
    //printf("  node %d has %d particles\n", tnode, t.num[tnode]);
    if (t.num[tnode] <= blockSize) {
        // make the equivalent particles for this node
        (void) refineLeaf(p, t, t.ioffset[tnode], t.ioffset[tnode]+t.num[tnode]);
        //for (int i=t.ioffset[tnode]; i<t.ioffset[tnode]+t.num[tnode]; ++i)
        //    printf("  %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);
    } else {
        // recurse and check child nodes
        (void) refineTree(p, t, 2*tnode);
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
void calcEquivalents(Parts& p, Parts& ep, Tree& t, int tnode) {
    //printf("  node %d has %d particles\n", tnode, t.num[tnode]);

    t.epoffset[tnode] = tnode * blockSize;
    t.epnum[tnode] = 0;
    //printf("    equivalent particles start at %d\n", t.epoffset[tnode]);

    // loop over children, adding equivalent particles to our list
    for (auto ichild = 2*tnode; ichild < 2*tnode+2; ++ichild) {
        //printf("  child %d has %d particles\n", ichild, t.num[ichild]);

        // split on whether this child is a leaf node or not
        if (t.num[ichild] > blockSize) {
            // this child is a non-leaf node and needs to make equivalent particles
            (void) calcEquivalents(p, ep, t, ichild);

            //printf("  back in node %d...\n", tnode);

            // now we read those equivalent particles and make higher-level equivalents
            //printf("    child %d made equiv parts %d to %d\n", ichild, t.epoffset[ichild], t.epoffset[ichild]+t.epnum[ichild]);

            // merge pairs of child's equivalent particles until we have half
            int numEqps = (t.epnum[ichild]+1) / 2;
            int istart = (blockSize/2) * ichild;
            int istop = istart + numEqps;
            //printf("    making %d equivalent particles %d to %d\n", numEqps, istart, istop);

            // loop over new equivalent particles and real particles together
            int iep = istart;
            int ip = t.epoffset[ichild] + 1;
            for (; iep<istop and ip<t.epoffset[ichild]+t.epnum[ichild];
                   iep++,     ip+=2) {
                //printf("    merging %d and %d into %d\n", ip-1,ip,iep);
                float pairm = ep.m[ip-1] + ep.m[ip];
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
            int numEqps = (t.num[ichild]+1) / 2;
            int istart = (blockSize/2) * ichild;
            int istop = istart + numEqps;
            //printf("    making %d equivalent particles %d to %d\n", numEqps, istart, istop);

            // loop over new equivalent particles and real particles together
            int iep = istart;
            int ip = t.ioffset[ichild] + 1;
            for (; iep<istop and ip<t.ioffset[ichild]+t.num[ichild];
                   iep++,     ip+=2) {
                //printf("    merging %d and %d into %d\n", ip-1,ip,iep);
                float pairm = p.m[ip-1] + p.m[ip];
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

    //printf("  node %d finally has %d equivalent particles\n", tnode, t.epnum[tnode]);
}

//
// basic usage
//
static void usage() {
    fprintf(stderr, "Usage: onbody [-n=<factor>] [iterations]\n");
    exit(1);
}

//
// main routine - run the program
//
int main(int argc, char *argv[]) {

    static std::vector<int> test_iterations = {10, 5, 2};
    int numSrcs = 10000;
    int numTargs = 10000;

    if (argc > 1) {
        if (strncmp(argv[1], "-n=", 3) == 0) {
            int num = atof(argv[1] + 3);
            if (num < 1) usage();
            numSrcs = num;
            numTargs = num;
        }
    }

    printf("Allocate and initialize\n");
    reset_and_start_timer();

    // allocate space for sources and targets
    Parts srcs;
    srcs.n = numSrcs;
    srcs.x.resize(srcs.n);
    srcs.y.resize(srcs.n);
    srcs.z.resize(srcs.n);
    srcs.r.resize(srcs.n);
    srcs.m.resize(srcs.n);
    srcs.itemp.resize(srcs.n);
    srcs.ftemp.resize(srcs.n);

    Parts targs;
    targs.n = numTargs;
    targs.x.resize(targs.n);
    targs.y.resize(targs.n);
    targs.z.resize(targs.n);
    targs.u.resize(targs.n);
    targs.v.resize(targs.n);
    targs.w.resize(targs.n);
    targs.itemp.resize(targs.n);
    targs.ftemp.resize(targs.n);

    // initialize particle data
    for (auto&& x : srcs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : srcs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : srcs.z) { z = (float)rand()/(float)RAND_MAX; }
    for (auto&& r : srcs.r) { r = 1.0f / cbrt((float)srcs.n); }
    for (auto&& m : srcs.m) { m = 2.0f*(float)rand()/(float)RAND_MAX / (float)srcs.n; }

    for (auto&& x : targs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : targs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : targs.z) { z = (float)rand()/(float)RAND_MAX; }
    printf("  init parts time:\t[%.3f] million cycles\n", get_elapsed_mcycles());


    // allocate and initialize tree
    printf("\nBuilding the source tree\n");
    reset_and_start_timer();
    Tree stree;
    printf("  with %d particles and block size of %d\n", numSrcs, blockSize);
    uint32_t numLeaf = 1 + ((numSrcs-1)/blockSize);
    printf("  %d nodes at leaf level\n", numLeaf);
    stree.levels = 1 + log_2(2*numLeaf-1);
    printf("  makes %d levels in tree\n", stree.levels);
    stree.numnodes = 1 << stree.levels;
    printf("  and %d total nodes in tree\n", stree.numnodes);
    stree.x.resize(stree.numnodes);
    stree.y.resize(stree.numnodes);
    stree.z.resize(stree.numnodes);
    stree.s.resize(stree.numnodes);
    stree.m.resize(stree.numnodes);
    stree.ioffset.resize(stree.numnodes);
    stree.num.resize(stree.numnodes);
    std::fill(stree.num.begin(), stree.num.end(), 0);
    printf("  allocate and init tree:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // split this node and recurse
    reset_and_start_timer();
    (void) splitNode(srcs, 0, srcs.n, stree, 1);
    printf("  build tree time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // find equivalent particles
    printf("\nCalculating equivalent particles\n");
    reset_and_start_timer();
    Parts eqpts;
    eqpts.n = (stree.numnodes/2) * blockSize;
    printf("  need %d particles\n", eqpts.n);
    eqpts.x.resize(eqpts.n);
    eqpts.y.resize(eqpts.n);
    eqpts.z.resize(eqpts.n);
    eqpts.r.resize(eqpts.n);
    eqpts.m.resize(eqpts.n);
    stree.epoffset.resize(stree.numnodes);
    stree.epnum.resize(stree.numnodes);
    printf("  allocate eqpts structures:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // first, reorder tree until all parts are adjacent in space-filling curve
    reset_and_start_timer();
    (void) refineTree(srcs, stree, 1);
    printf("  refine within leaf nodes:\t[%.3f] million cycles\n", get_elapsed_mcycles());
    //for (int i=0; i<stree.num[1]; ++i)
    //    printf("%d %g %g %g\n", i, srcs.x[i], srcs.y[i], srcs.z[i]);

    // then, march through arrays merging pairs as you go up
    reset_and_start_timer();
    (void) calcEquivalents(srcs, eqpts, stree, 1);
    printf("  create equivalent parts:\t[%.3f] million cycles\n", get_elapsed_mcycles());
    //printf("\n\n");
    //for (int ib=4; ib<8; ib++) {
    //    for (int i=stree.epoffset[ib]; i<stree.epoffset[ib]+stree.epnum[ib]; ++i) {
    //        printf("%d %g %g %g\n", i, eqpts.x[i], eqpts.y[i], eqpts.z[i]);
    //    }
    //}
    //printf("\n\n");
    //for (int ib=2; ib<4; ib++) {
    //    for (int i=stree.epoffset[ib]; i<stree.epoffset[ib]+stree.epnum[ib]; ++i) {
    //        printf("%d %g %g %g\n", i, eqpts.x[i], eqpts.y[i], eqpts.z[i]);
    //    }
    //}
    //printf("\n\n");
    //for (int ib=1; ib<2; ib++) {
    //    for (int i=stree.epoffset[ib]; i<stree.epoffset[ib]+stree.epnum[ib]; ++i) {
    //        printf("%d %g %g %g\n", i, eqpts.x[i], eqpts.y[i], eqpts.z[i]);
    //    }
    //}

    exit(0);

    // don't need the target tree for treecode, but will for fast code
    printf("\nBuilding the target tree\n");
    Tree ttree;
    printf("  with %d particles and block size of %d\n", numTargs, blockSize);

    //
    // Run the new O(N) equivalent particle method
    //
    //printf("\nRun the fast O(N) method\n");
    //double minFast = 1e30;
    //for (unsigned int i = 0; i < test_iterations[0]; ++i) {
    //    reset_and_start_timer();
    //    nbody_fastsumm(srcs, targs);
    //    double dt = get_elapsed_mcycles();
    //    printf("  this run time:\t\t\t[%.3f] million cycles\n", dt);
    //    minFast = std::min(minFast, dt);
    //}
    //printf("[onbody fast]:\t\t[%.3f] million cycles\n", minFast);
    // write sample results
    //for (int i = 0; i < 4; i++) printf("   particle %d vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);
    // save the results for comparison

    //
    // Run a simple O(NlogN) treecode
    //
    printf("\nRun the treecode O(NlogN)\n");
    double minTreecode = 1e30;
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        nbody_treecode1(srcs, stree, targs, 3.0f);
        double dt = get_elapsed_mcycles();
        printf("  this run time:\t\t\t[%.3f] million cycles\n", dt);
        minTreecode = std::min(minTreecode, dt);
    }
    printf("[onbody treecode]:\t\t[%.3f] million cycles\n", minTreecode);
    // write sample results
    for (int i = 0; i < 4; i++) printf("   particle %d vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);
    // save the results for comparison
    std::vector<float> treecodeu = targs.u;

    //
    // Run the O(N^2) implementation
    //
    printf("\nRun the naive O(N^2) method\n");
    double minNaive = 1e30;
    for (unsigned int i = 0; i < test_iterations[2]; ++i) {
        reset_and_start_timer();
        nbody_naive(srcs, targs);
        double dt = get_elapsed_mcycles();
        printf("  this run time:\t\t\t[%.3f] million cycles\n", dt);
        minNaive = std::min(minNaive, dt);
    }
    printf("[onbody naive]:\t\t[%.3f] million cycles\n", minNaive);
    // write sample results
    for (int i = 0; i < 4; i++) printf("   particle %d vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);

    // compare accuracy
    float errsum = 0.0;
    for (auto i=0; i< targs.u.size(); ++i) {
        float thiserr = treecodeu[i]-targs.u[i];
        errsum += thiserr*thiserr;
    }
    printf("\nRMS error in treecode is %g\n", sqrtf(errsum/targs.u.size()));

    return 0;
}
