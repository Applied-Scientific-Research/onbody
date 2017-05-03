/*
 * onbody - testbed for an O(N) 3d gravitational solver
 *
 * Copyright (c) 2017, Mark J Stock
 */

#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "timing.h"


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


static void usage() {
    fprintf(stderr, "Usage: onbody [-n=<factor>] [iterations]\n");
    exit(1);
}

int main(int argc, char *argv[]) {

    static unsigned int test_iterations = 4;
    int numSrcs = 10000;
    int numTargs = 10000;
    const int maxGangSize = 64;

    if (argc > 1) {
        if (strncmp(argv[1], "-n=", 3) == 0) {
            int num = atof(argv[1] + 3);
            if (num < 1) usage();
            numSrcs = maxGangSize*(num/maxGangSize);
            numTargs = maxGangSize*(num/maxGangSize);
        }
    }
    if ((argc == 3) || (argc == 4)) {
        test_iterations = atoi(argv[argc - 2]);
    }


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
    for (int i = 0; i < numSrcs; i++) {
        sx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sy[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sz[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        sr[i] = 1.0 / sqrt((float)numSrcs);
        sm[i] = 1.0 / sqrt((float)numSrcs);
    }
    for (int i = 0; i < numTargs; i++) {
        tx[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        ty[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
        tz[i] = 2.*(float)rand()/(float)RAND_MAX - 1.0;
    }

    // make a tree

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
