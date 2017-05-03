/*
 * onbody - testbed for an O(N) 3d gravitational solver
 *
 * Copyright (c) 2017, Mark J Stock
 */

#include <cstdlib>
#include <stdio.h>
#include <string.h>
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
};

//
// A tree, made of a structure of arrays
// root node is 1, children are 2,3, their children 4,5 and 6,7
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

//
// Caller for the scalar kernel
//
void nbody(Parts& srcs, Parts& targs) {

    #pragma omp parallel for
    for (int i = 0; i < targs.n; i++) {
        targs.u[i] = 0.0;
        targs.v[i] = 0.0;
        targs.w[i] = 0.0;
        for (int j = 0; j < srcs.n; j++) {
            nbody_kernel(srcs.x[j], srcs.y[j], srcs.z[j], srcs.r[j], srcs.m[j],
                         targs.x[i], targs.y[i], targs.z[i],
                         targs.u[i], targs.v[i], targs.w[i]);
        }
    }
}

//
// Sort but retain only sorted index! Uses C++11 lambdas
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
}

//
// Split this segment of the particles on its longest axis
//
void splitNode(Parts& p, size_t begin, size_t end, Tree& t, int tnode) {

    printf("splitNode %d  %ld %ld\n", tnode, begin, end);

    // debug print - starting condition
    for (int i=begin; i<begin+10 and i<end; ++i)
        printf("  node %i %g %g %g\n", i, p.x[i], p.y[i], p.z[i]);

    // find the min/max of the three axes
    printf("find min/max\n");
    reset_and_start_timer();
    std::vector<float> boxsizes(3);
    auto minmax = minMaxValue(p.x, begin, end);
    boxsizes[0] = minmax.second - minmax.first;
    printf("  node x min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.y, begin, end);
    boxsizes[1] = minmax.second - minmax.first;
    printf("       y min/max %g %g\n", minmax.first, minmax.second);
    minmax = minMaxValue(p.z, begin, end);
    boxsizes[2] = minmax.second - minmax.first;
    printf("       z min/max %g %g\n", minmax.first, minmax.second);
    // and find longest box edge
    auto maxaxis = std::max_element(boxsizes.begin(), boxsizes.end()) - boxsizes.begin();
    printf("  longest axis is %ld\n", maxaxis);
    printf("  minmax time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // sort it
    printf("sort\n");
    reset_and_start_timer();
    auto idx = sortIndexes(p.x);
    for (int i=begin; i<begin+10 and i<end; ++i) printf("  node %d %ld %g\n", i, idx[i], p.x[idx[i]]);
    printf("  sort time:\t[%.3f] million cycles\n", get_elapsed_mcycles());

    // rearrange the elements
    printf("reorder\n");
    reset_and_start_timer();
    // make a temporary vector
    std::vector<float> temp(end-begin);

    printf("  reorder time:\t[%.3f] million cycles\n", get_elapsed_mcycles());
}

//
// make a VAMsplit k-d tree from this set of particles
//
Tree makeTree(Parts& p) {

    printf("Building the tree\n");

    // initialize the tree
    Tree tree;

    // set up the root of the tree (node number 1, not 0!)
    int inode = 1;

    // split this node
    (void) splitNode(p, 0, 256, tree, inode);

    return tree;
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

    static unsigned int test_iterations = 4;
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
    if ((argc == 3) || (argc == 4)) {
        test_iterations = atoi(argv[argc - 2]);
    }

    numSrcs = blockSize*(numSrcs/blockSize);
    numTargs = blockSize*(numTargs/blockSize);

    // allocate space for sources and targets
    Parts srcs;
    srcs.n = numSrcs;
    srcs.x.resize(srcs.n);
    srcs.y.resize(srcs.n);
    srcs.z.resize(srcs.n);
    srcs.r.resize(srcs.n);
    srcs.m.resize(srcs.n);

    Parts targs;
    targs.n = numTargs;
    targs.x.resize(targs.n);
    targs.y.resize(targs.n);
    targs.z.resize(targs.n);
    targs.u.resize(targs.n);
    targs.v.resize(targs.n);
    targs.w.resize(targs.n);

    // initialize particle data
    for (auto&& x : srcs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : srcs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : srcs.z) { z = (float)rand()/(float)RAND_MAX; }
    for (auto&& r : srcs.r) { r = 1.0 / cbrt((float)srcs.n); }
    for (auto&& m : srcs.m) { m = 2.*(float)rand()/(float)RAND_MAX / (float)srcs.n; }

    for (auto&& x : targs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : targs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : targs.z) { z = (float)rand()/(float)RAND_MAX; }

    // make a tree
    Tree stree = makeTree(srcs);

    //
    // Run the implementation a few times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations; ++i) {
        reset_and_start_timer();
        nbody(srcs, targs);
        double dt = get_elapsed_mcycles();
        printf("time of base run:\t\t\t[%.3f] million cycles\n", dt);
        minSerial = std::min(minSerial, dt);
    }

    printf("[onbody]:\t\t[%.3f] million cycles\n", minSerial);

    // Write sample results
    for (int i = 0; i < 4; i++) printf("   particle %d vel %g %g %g\n",i,targs.u[i],targs.v[i],targs.w[i]);

    return 0;
}
