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
    std::vector<float> t;
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
    // node masses
    std::vector<float> m;
    // point offset and count
    std::vector<int> ioffset;
    std::vector<int> num;
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
void nbody_naive(Parts& srcs, Parts& targs) {
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
// Recursive kernel for the treecode
//
void nbody_by_targ(Parts& srcs, Tree& stree, int tnode, float theta,
                   const float tx, const float ty, const float tz,
                   float& tax, float& tay, float& taz) {

    // distance from box center of mass to target point
    const float dx = stree.x[tnode] - tx;
    const float dy = stree.y[tnode] - ty;
    const float dz = stree.z[tnode] - tz;
    const float dist = sqrtf(dx*dx + dy*dy + dz*dz);

    // is source tree node far enough away?
    if (dist / stree.s[tnode] > theta) {
        // box is far enough removed, approximate its influence
        nbody_kernel(stree.x[tnode], stree.y[tnode], stree.z[tnode], 0.0f, stree.m[tnode],
                     tx, ty, tz, tax, tay, taz);
    } else if (stree.num[tnode] < blockSize) {
        // box is too close and is a leaf node, look at individual particles
        //nbody_src_block()
    } else {
        // box is too close, open up its children
        (void) nbody_by_targ(srcs, stree, 2*tnode,   theta, tx, ty, tz, tax, tay, taz);
        (void) nbody_by_targ(srcs, stree, 2*tnode+1, theta, tx, ty, tz, tax, tay, taz);
    }
}

//
// Caller for the O(NlogN) kernel
//
void nbody_treecode(Parts& srcs, Tree& stree, Parts& targs, float theta) {
    #pragma omp parallel for
    for (int i = 0; i < targs.n; i++) {
        targs.u[i] = 0.0f;
        targs.v[i] = 0.0f;
        targs.w[i] = 0.0f;
        nbody_by_targ(srcs, stree, 1, theta,
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
// Make a VAMsplit k-d tree from this set of particles
// Split this segment of the particles on its longest axis
//
void splitNode(Parts& p, size_t pfirst, size_t plast, Tree& t, int tnode) {

    printf("\nsplitNode %d  %ld %ld\n", tnode, pfirst, plast);

    // debug print - starting condition
    for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);

    // find the min/max of the three axes
    printf("find min/max\n");
    reset_and_start_timer();
    std::vector<float> boxsizes(3);
    auto minmax = minMaxValue(p.x, pfirst, plast);
    boxsizes[0] = minmax.second - minmax.first;
    printf("  node x min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.y, pfirst, plast);
    boxsizes[1] = minmax.second - minmax.first;
    printf("       y min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.z, pfirst, plast);
    boxsizes[2] = minmax.second - minmax.first;
    printf("       z min/max %g %g\n", minmax.first, minmax.second);
    printf("  minmax time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // find total mass and center of mass
    printf("find mass/cm\n");
    reset_and_start_timer();
    t.m[tnode] = std::accumulate(p.m.begin()+pfirst, p.m.begin()+plast, 0.0);
    t.x[tnode] = std::inner_product(p.x.begin()+pfirst, p.x.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    t.y[tnode] = std::inner_product(p.y.begin()+pfirst, p.y.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    t.z[tnode] = std::inner_product(p.z.begin()+pfirst, p.z.begin()+plast, p.m.begin()+pfirst, 0.0) / t.m[tnode];
    printf("  total mass %g and cm %g %g %g\n", t.m[tnode], t.x[tnode], t.y[tnode], t.z[tnode]);

    // write all this data to the tree node
    t.ioffset[tnode] = pfirst;
    t.num[tnode] = plast - pfirst;
    printf("  tree node has offset %d and num %d\n", t.ioffset[tnode], t.num[tnode]);

    // find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();
    printf("  longest axis is %ld, length %g\n", maxaxis, boxsizes[maxaxis]);
    t.s[tnode] = boxsizes[maxaxis];
    printf("  tree node time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // no need to split or compute further
    if (t.num[tnode] <= blockSize) return;

    // sort it along the big axis
    printf("sort\n");
    reset_and_start_timer();
    auto idx = sortIndexes(p.x);
    for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %ld %g\n", i, idx[i], p.x[idx[i]]);
    printf("  sort time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // rearrange the elements
    printf("reorder\n");
    reset_and_start_timer();
    // copy this segment into the temp array
    std::copy(p.x.begin()+pfirst, p.x.end()+plast, p.t.begin()+pfirst);
    //for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  temp %d %g\n", i, p.t[i]);
    // now write back to the original array using the indexes from the sort
    for (int i=pfirst; i<plast; ++i) p.x[i] = p.t[idx[i]];
    // redo for the other axes
    std::copy(p.y.begin()+pfirst, p.y.end()+plast, p.t.begin()+pfirst);
    for (int i=pfirst; i<plast; ++i) p.y[i] = p.t[idx[i]];
    std::copy(p.z.begin()+pfirst, p.z.end()+plast, p.t.begin()+pfirst);
    for (int i=pfirst; i<plast; ++i) p.z[i] = p.t[idx[i]];
    std::copy(p.m.begin()+pfirst, p.m.end()+plast, p.t.begin()+pfirst);
    for (int i=pfirst; i<plast; ++i) p.m[i] = p.t[idx[i]];
    std::copy(p.r.begin()+pfirst, p.r.end()+plast, p.t.begin()+pfirst);
    for (int i=pfirst; i<plast; ++i) p.r[i] = p.t[idx[i]];
    // clean this up with an inline function
    // reorder(p.x, p.t, idx, pfirst, plast);
    for (int i=pfirst; i<pfirst+10 and i<plast; ++i) printf("  node %d %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);

    printf("  reorder time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // recursively call this routine for this node's new children

    // determine where the split should be
    size_t pmiddle = pfirst + blockSize * (1 << log_2((t.num[tnode]-1)/blockSize));
    printf("split at %ld %ld %ld into nodes %d %d\n", pfirst, pmiddle, plast, 2*tnode, 2*tnode+1);

    // call on the two children
    (void) splitNode(p, pfirst,  pmiddle, t, 2*tnode);
    (void) splitNode(p, pmiddle, plast,   t, 2*tnode+1);
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

    static std::vector<int> test_iterations = {10, 10, 4};
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

    //numSrcs = blockSize*(numSrcs/blockSize);
    //numTargs = blockSize*(numTargs/blockSize);

    // allocate space for sources and targets
    Parts srcs;
    srcs.n = numSrcs;
    srcs.x.resize(srcs.n);
    srcs.y.resize(srcs.n);
    srcs.z.resize(srcs.n);
    srcs.r.resize(srcs.n);
    srcs.m.resize(srcs.n);
    srcs.t.resize(srcs.n);

    Parts targs;
    targs.n = numTargs;
    targs.x.resize(targs.n);
    targs.y.resize(targs.n);
    targs.z.resize(targs.n);
    targs.u.resize(targs.n);
    targs.v.resize(targs.n);
    targs.w.resize(targs.n);
    targs.t.resize(targs.n);

    // initialize particle data
    for (auto&& x : srcs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : srcs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : srcs.z) { z = (float)rand()/(float)RAND_MAX; }
    for (auto&& r : srcs.r) { r = 1.0f / cbrt((float)srcs.n); }
    for (auto&& m : srcs.m) { m = 2.0f*(float)rand()/(float)RAND_MAX / (float)srcs.n; }

    for (auto&& x : targs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : targs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : targs.z) { z = (float)rand()/(float)RAND_MAX; }


    // allocate and initialize tree
    printf("\nBuilding the source tree\n");
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

    // split this node and recurse
    (void) splitNode(srcs, 0, srcs.n, stree, 1);

    // find equivalent particles

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
    //printf("\nRun the treecode O(NlogN)\n");
    //double minTreecode = 1e30;
    //for (unsigned int i = 0; i < test_iterations[1]; ++i) {
    //    reset_and_start_timer();
    //    nbody_treecode(srcs, targs);
    //    double dt = get_elapsed_mcycles();
    //    printf("  this run time:\t\t\t[%.3f] million cycles\n", dt);
    //    minTreecode = std::min(minTreecode, dt);
    //}
    //printf("[onbody treecode]:\t\t[%.3f] million cycles\n", minTreecode);
    // write sample results
    //for (int i = 0; i < 4; i++) printf("   particle %d vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);
    // save the results for comparison

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

    return 0;
}
