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
#include <algorithm>
#include "timing.h"

const int blockSize = 64;

//
// A set of particles, can be sources or targets
//
struct Parts {
    int n;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> r;
    std::vector<float> m;
};

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
                                float* tax, float* tay, float* taz) {
    // 19 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    const float dz = sz - tz;
    float r2 = dx*dx + dy*dy + dz*dz + sr*sr;
    r2 = sm/(r2*sqrt(r2));
    (*tax) += r2 * dx;
    (*tay) += r2 * dy;
    (*taz) += r2 * dz;
}

//
// A blocked kernel
//

//
// Caller for the scalar kernel
//
void nbody(int numSrcs, float sx[], float sy[], float sz[], float sr[], float sm[],
           int numTarg, float tx[], float ty[], float tz[],
                        float tax[], float tay[], float taz[])
{
    #pragma omp parallel for
    for (int i = 0; i < numTarg; i++) {
        tax[i] = 0.0;
        tay[i] = 0.0;
        taz[i] = 0.0;
        for (int j = 0; j < numSrcs; j++) {
            nbody_kernel(sx[j], sy[j], sz[j], sr[j], sm[j],
                         tx[i], ty[i], tz[i],
                         &tax[i], &tay[i], &taz[i]);
        }
    }
}

//
// Split this segment of the particles on its longest axis
//
void splitNode(std::vector<float> x) {

    // find the min/max of the three axes
    //float xmin = std::min_element(std::begin(x), std::end(x));
    //std::pair<float,float> xrange = std::minmax_element(x.begin(), x.end());
    auto xrange = std::minmax_element(x.begin(), x.end());

    for (int i=0; i<10; ++i) printf("  node %i %g\n", i, x[i]);
    printf("  node x min/max %g %g\n", x[xrange.first-x.begin()], x[xrange.second-x.begin()]);
}

//
// make a VAMsplit k-d tree from this set of particles
//
Tree makeTree(Parts p) {

    printf("Building the tree\n");

    // initialize the tree
    Tree tree;

    // set up the root of the tree (node number 1, not 0!)
    int inode = 1;

    // split this node
    (void) splitNode(p.x);

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

    // allocate particle data
    float *sx = new float[numSrcs];
    float *sy = new float[numSrcs];
    float *sz = new float[numSrcs];
    float *sr = new float[numSrcs];
    float *sm = new float[numSrcs];

    float *tx = new float[numTargs];
    float *ty = new float[numTargs];
    float *tz = new float[numTargs];
    float *tax = new float[numTargs];
    float *tay = new float[numTargs];
    float *taz = new float[numTargs];

    // initialize particle data
    for (auto&& x : srcs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : srcs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : srcs.z) { z = (float)rand()/(float)RAND_MAX; }
    for (auto&& r : srcs.r) { r = 1.0 / cbrt((float)srcs.n); }
    for (auto&& m : srcs.m) { m = 2.*(float)rand()/(float)RAND_MAX / (float)srcs.n; }

    for (auto&& x : targs.x) { x = (float)rand()/(float)RAND_MAX; }
    for (auto&& y : targs.y) { y = (float)rand()/(float)RAND_MAX; }
    for (auto&& z : targs.z) { z = (float)rand()/(float)RAND_MAX; }

    for (int i = 0; i < numSrcs; i++) {
        sx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sy[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sz[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        // really should be cube root
        sr[i] = 1.0 / sqrt((float)numSrcs);
        sm[i] = 2.0 * ((float)rand()/(float)RAND_MAX) / (float)numSrcs;
    }
    for (int i = 0; i < numTargs; i++) {
        tx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ty[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        tz[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
    }

    // make a tree
    Tree stree = makeTree(srcs);

    //
    // Run the implementation a few times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (unsigned int i = 0; i < test_iterations; ++i) {
        reset_and_start_timer();
        nbody(numSrcs, sx, sy, sz, sr, sm,
              numTargs, tx, ty, tz, tax, tay, taz);
        double dt = get_elapsed_mcycles();
        printf("@time of base run:\t\t\t[%.3f] million cycles\n", dt);
        minSerial = std::min(minSerial, dt);
    }

    printf("[onbody]:\t\t[%.3f] million cycles\n", minSerial);

    // Write sample results
    for (int i = 0; i < 4; i++) printf("   particle %d vel %g %g %g\n",i,tax[i],tay[i],taz[i]);

    return 0;
}
