/*
 * BarycentricLagrange.hpp - functions supporting barycentric Lagrange interpolation
 *
 * Copyright (c) 2022, Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#include "Parts.hpp"
#include "Tree.hpp"

const int32_t maxorder = 15;

// "global" variables, the locations of the Chebyshev nodes of the 2nd kind
template <class S>
std::array<S,maxorder+1> sk;

template <class S>
std::array<S,maxorder+1> wk;

// generate the sk locations [-1..1]
template <class S>
void set_sk(const int32_t _n) {
    assert(_n > 0 && "ERROR (set_sk): n must be >0");
    assert(_n <= maxorder && "ERROR (set_sk): n is too high!");
    for (int32_t k=0; k<=_n; ++k) {
        sk<S>[k] = std::cos(k*M_PI/_n);
    }
}

// generate the wk weights
template <class S>
void set_wk(const int32_t _n) {
    assert(_n > 0 && "ERROR (set_wk): n must be >0");
    assert(_n <= maxorder && "ERROR (set_wk): n is too high!");
    wk<S>[0] = 0.5;
    for (int32_t k=1; k<_n; ++k) {
        wk<S>[k] = 1.;
    }
    wk<S>[_n] = 0.5;
}


//
// Loop over all nodes in the tree and calculate the n^d barycentric equivalent particles
//
template <class S, class A, int PD, int SD, int OD>
void calcBarycentricLagrange(Parts<S,A,PD,SD,OD>& p, Parts<S,A,PD,SD,OD>& ep, Tree<S,PD,SD>& t, size_t tnode) {
    printf("  node %ld has %ld particles\n", tnode, t.num[tnode]);
    if (not p.are_sources or not ep.are_sources) return;

    t.epoffset[tnode] = tnode * blockSize;
    t.epnum[tnode] = 0;
    printf("    equivalent particles start at %ld\n", t.epoffset[tnode]);

    // map the Chebyshev nodes to this cluster's bounds
    // note that t.x[d][tnode] is the center of mass - not the center of the cluster!!!
    // the cluster size is t.ns[d][tnode]
    // geometric center is t.nc[d][tnode]

    // loop over children, adding equivalent particles to our list
    for (size_t ichild = 2*tnode; ichild < 2*tnode+2; ++ichild) {
        printf("  child %ld has %ld particles\n", ichild, t.num[ichild]);

        // split on whether this child is a leaf node or not
        if (t.num[ichild] > blockSize) {
            // this child is a non-leaf node and needs to make equivalent particles
            (void) calcBarycentricLagrange(p, ep, t, ichild);

            printf("  back in node %ld...\n", tnode);
            exit(1);

            // now we read those equivalent particles and make higher-level equivalents
            //printf("    child %ld made equiv parts %ld to %ld\n", ichild, t.epoffset[ichild], t.epoffset[ichild]+t.epnum[ichild]);

        } else {
            // this child is a leaf node
            //printf("    child leaf node has particles %ld to %ld\n", t.ioffset[ichild], t.ioffset[ichild]+t.num[ichild]);

            // if we're a leaf node, merge pairs of particles until we have half
            size_t numEqps = (t.num[ichild]+1) / 2;
            size_t istart = (blockSize/2) * ichild;
            size_t istop = istart + numEqps;
            printf("    making %ld equivalent particles %ld to %ld\n", numEqps, istart, istop);

            t.epnum[tnode] += numEqps;
        }
    }

    //printf("  node %d finally has %d equivalent particles, offset %d\n", tnode, t.epnum[tnode], t.epoffset[tnode]);

}
