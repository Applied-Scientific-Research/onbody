/*
 * BarycentricLagrange.hpp - functions supporting barycentric Lagrange interpolation
 *
 * Copyright (c) 2022, Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#include "MathHelper.hpp"
#include "Parts.hpp"
#include "Tree.hpp"

#include <cassert>

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
        // using negative so that numbers start low and go high
        sk<S>[k] = -std::cos(k*M_PI/_n);
    }
}

// generate the wk weights
template <class S>
void set_wk(const int32_t _n) {
    assert(_n > 0 && "ERROR (set_wk): n must be >0");
    assert(_n <= maxorder && "ERROR (set_wk): n is too high!");
    wk<S>[0] = 0.5*ipow(-1,0);
    for (int32_t k=1; k<_n; ++k) {
        wk<S>[k] = 1.*ipow(-1,k);
    }
    wk<S>[_n] = 0.5*ipow(-1,_n);
}


//
// Loop over all nodes in the tree and calculate the n^d barycentric equivalent particles
//
template <class S, class A, int PD, int SD, int OD>
void calcBarycentricLagrange(Parts<S,A,PD,SD,OD>& p,
                             Parts<S,A,PD,SD,OD>& ep,
                             Tree<S,PD,SD>& t,
                             const int32_t order,
                             const size_t tnode) {

    const bool dbg = false;
    const bool interp_radii = true;

    if (dbg) printf("  node %ld has %ld particles\n", tnode, t.num[tnode]);
    if (not p.are_sources or not ep.are_sources) return;

    // set the locations and weights of the barycentric particles
    if (tnode == 1) {
        (void) set_sk<S>(order);
        (void) set_wk<S>(order);
    }

    // set some constants
    const size_t ncp = order+1;		// number of Chebyshev points in each direction
    const size_t numEqps = ipow<size_t>(ncp,PD);
    //printf("    ncp %ld and numEqps %ld\n", ncp, numEqps);
    assert(numEqps <= blockSize && "ERROR (calcBarycentricLagrange): requested order too large for blockSize");

    // continue using blockSize for offset
    t.epoffset[tnode] = tnode * blockSize;
    t.epnum[tnode] = 0;
    const size_t iepstart = t.epoffset[tnode];
    const size_t iepstop = iepstart + numEqps;
    if (dbg) printf("    equivalent particles start at %ld\n", iepstart);

    // map the Chebyshev nodes to this cluster's bounds

    // make a local copy of the sk coordinates
    //S lsk[PD][ncp];
    std::vector<S> lsk(PD*ncp);
    {
        auto lsk_iter = std::begin(lsk);
        for (size_t d=0; d<PD; ++d) {
            for (size_t k=0; k<ncp; ++k) {
                //lsk[d][k] = t.nc[d][tnode] + 0.5 * sk<S>[k] * t.ns[d][tnode];
                *lsk_iter = t.nc[d][tnode] + 0.5 * sk<S>[k] * t.ns[d][tnode];
                ++lsk_iter;
            }
        }
    }

    // note that t.x[d][tnode] is the center of mass - not the center of the cluster!!!
    // the cluster size is t.ns[d][tnode]
    // geometric center is t.nc[d][tnode]
    for (size_t d=0; d<PD; ++d) {
        const size_t divisor = ipow<size_t>(ncp,d);
        //printf("    d %ld and divisor %ld\n", d, divisor);
        for (size_t i=0; i<numEqps; ++i) {
            ep.x[d][iepstart+i] = t.nc[d][tnode] + 0.5 * sk<S>[(i/divisor)%ncp] * t.ns[d][tnode];
        }
    }

    // and locate the remainder of the particles (unused) to the cell center
    for (size_t i=iepstop; i<iepstart+blockSize; ++i) {
        for (size_t d=0; d<PD; ++d) ep.x[d][i] = t.nc[d][tnode];
    }

    if (dbg) for (size_t i=iepstart; i<iepstop; ++i) printf("    eq part %ld is at %g %g %g\n", i, ep.x[0][i], ep.x[1][i], ep.x[2][i]);
    //for (size_t i=iepstart; i<iepstart+blockSize; ++i) printf("    eq part %ld is at %g %g %g\n", i, ep.x[0][i], ep.x[1][i], ep.x[2][i]);

    // set all equiv. particles weights
    for (size_t i=iepstart; i<iepstart+blockSize; ++i) {
        for (size_t d=0; d<SD; ++d) ep.s[d][i] = 0.0;
    }

    // initialize radii to zero, or just copy the first particle radius
    if (interp_radii) {
        for (size_t i=iepstart; i<iepstart+blockSize; ++i) ep.r[i] = 0.0;
    } else {
        for (size_t i=iepstart; i<iepstart+blockSize; ++i) ep.r[i] = p.r[t.ioffset[tnode]];
    }

    // store a sum of weights
    std::vector<S> wgtsum(blockSize, 0.0);

    // precompute these useful indices
    std::vector<std::array<S,PD>> kidx;
    kidx.resize(numEqps);
    for (size_t d=0; d<PD; ++d) {
        const size_t divisor = ipow<size_t>(ncp,d);
        for (size_t i=0; i<numEqps; ++i) {
            kidx[i][d] = (i/divisor) % ncp;
        }
    }

    // loop over children, adding equivalent particles to our list
    for (size_t ichild = 2*tnode; ichild < 2*tnode+2; ++ichild) {
        if (dbg) printf("  child %ld has %ld particles\n", ichild, t.num[ichild]);

        // split on whether this child is a leaf node or not
        if (t.num[ichild] > blockSize) {

            // not a leaf node
            if (dbg) printf("    from %ld to %ld\n", t.ioffset[ichild], t.ioffset[ichild]+t.num[ichild]);

            // this child is a non-leaf node and needs to make equivalent particles
            #pragma omp task shared(p,ep,t)
            (void) calcBarycentricLagrange(p, ep, t, order, ichild);
            // need to call both children here to gain any parallelism

            if (dbg) printf("  back in node %ld...\n", tnode);

            // now we read those equivalent particles and make higher-level equivalents

            // here istart and istop are the previous equivalent particles
            const size_t istart = t.epoffset[ichild];
            const size_t istop = istart + t.epnum[ichild];
            if (dbg) printf("    starting with equiv particles %ld to %ld\n", istart, istop);

            // and here is the range of new equivalent particles
            if (dbg) printf("    putting into %ld new equiv particles %ld to %ld\n", numEqps, iepstart, iepstop);

            // generate a reusable 2D array
            std::array<std::vector<S>,PD> amat;
            for (size_t d=0; d<PD; ++d) amat[d].resize(ncp);

            // now do the work
            for (size_t ip=istart; ip<istop; ++ip) {
                //printf("      child part %ld at %g %g %g mass %g\n", ip, p.x[0][ip], p.x[1][ip], p.x[2][ip], p.s[0][ip]);

                int32_t flag[PD];
                for (size_t d=0; d<PD; ++d) flag[d] = -1;
                S sum[PD];
                for (size_t d=0; d<PD; ++d) sum[d] = 0.0;
                for (size_t d=0; d<PD; ++d) for (size_t k=0; k<ncp; ++k) amat[d][k] = 0.0;

                // loop over coord indices and Cheby points to compute a
                auto lsk_iter = std::begin(lsk);
                for (size_t d=0; d<PD; ++d) {
                for (size_t k=0; k<ncp; ++k) {
                    // note the use of ep here!
                    //const S dist = ep.x[d][ip] - lsk[d][k];
                    const S dist = ep.x[d][ip] - *lsk_iter;
                    ++lsk_iter;
                    if (std::abs(dist) < 1.e-10) {
                        flag[d] = k;
                    } else {
                        amat[d][k] = wk<S>[k] / dist;
                        sum[d] += amat[d][k];
                    }
                }
                }

                // if a flag was set, remove singularity
                for (size_t d=0; d<PD; ++d) {
                    if (flag[d] > -1) {
                        sum[d] = 1.0;
                        for (size_t k=0; k<ncp; ++k) amat[d][k] = 0.0;
                        amat[d][flag[d]] = 1.0;
                    }
                }

                // can compute denominator now
                S denom = 1.0;
                for (size_t d=0; d<PD; ++d) denom *= sum[d];
                denom = 1.0/denom;

                // final loop to accumulate into equivalent weights
                for (size_t i=0; i<numEqps; ++i) {
                    const size_t iep = iepstart + i;
                    S wgt = denom;
                    for (size_t d=0; d<PD; ++d) {
                        //const size_t k = (i/ipow<size_t>(ncp,d)) % ncp;
                        //wgt *= amat[d][k];
                        wgt *= amat[d][kidx[i][d]];
                    }
                    // note the use of ep for both here!
                    for (size_t d=0; d<SD; ++d) ep.s[d][iep] += wgt * ep.s[d][ip];

                    if (interp_radii) {
                        ep.r[iep] += std::abs(wgt) * ep.r[ip];
                        wgtsum[i] += std::abs(wgt);
                    }
                }
            }

            // now adjust particle radii
            //for (size_t i=0; i<numEqps; ++i) ep.r[iepstart+i] /= wgtsum[i];

            //if (dbg) for (size_t i=iepstart; i<iepstop; ++i) printf("    eq part %ld is at %g %g %g mass %g rad %g\n", i, ep.x[0][i], ep.x[1][i], ep.x[2][i], ep.s[0][i], ep.r[i]);

            t.epnum[tnode] = numEqps;

        } else {

            // this child is a leaf node
            const size_t istart = t.ioffset[ichild];
            const size_t istop = istart + t.num[ichild];
            if (dbg) printf("    child leaf node has particles %ld to %ld\n", istart, istop);

            // and here is the range of equivalent particles
            if (dbg) printf("    putting into %ld equivalent particles %ld to %ld\n", numEqps, iepstart, iepstop);

            // generate a reusable 2D array
            std::array<std::vector<S>,PD> amat;
            for (size_t d=0; d<PD; ++d) amat[d].resize(ncp);

            // now do the work
            for (size_t ip=istart; ip<istop; ++ip) {
                //printf("      child part %ld at %g %g %g mass %g\n", ip, p.x[0][ip], p.x[1][ip], p.x[2][ip], p.s[0][ip]);

                int32_t flag[PD];
                for (size_t d=0; d<PD; ++d) flag[d] = -1;
                S sum[PD];
                for (size_t d=0; d<PD; ++d) sum[d] = 0.0;
                for (size_t d=0; d<PD; ++d) for (size_t k=0; k<ncp; ++k) amat[d][k] = 0.0;

                // loop over coord indices and Cheby points to compute a
                auto lsk_iter = std::begin(lsk);
                for (size_t d=0; d<PD; ++d) {
                for (size_t k=0; k<ncp; ++k) {
                    //const S dist = p.x[d][ip] - lsk[d][k];
                    const S dist = p.x[d][ip] - *lsk_iter;
                    ++lsk_iter;
                    if (std::abs(dist) < 1.e-10) {
                        flag[d] = k;
                    } else {
                        amat[d][k] = wk<S>[k] / dist;
                        sum[d] += amat[d][k];
                    }
                    //printf("      d %ld k %ld x %g lsk %g flag %d amat %g\n", d, k, p.x[d][ip], lsk[d][k], flag[d], amat[d][k]);
                }
                }

                // if a flag was set, remove singularity
                for (size_t d=0; d<PD; ++d) {
                    if (flag[d] > -1) {
                        sum[d] = 1.0;
                        for (size_t k=0; k<ncp; ++k) amat[d][k] = 0.0;
                        amat[d][flag[d]] = 1.0;
                    }
                }

                // can compute denominator now
                S denom = 1.0;
                for (size_t d=0; d<PD; ++d) denom *= sum[d];
                denom = 1.0/denom;

                // final loop to accumulate into equivalent weights
                for (size_t i=0; i<numEqps; ++i) {
                    const size_t iep = iepstart + i;
                    S wgt = denom;
                    for (size_t d=0; d<PD; ++d) {
                        //const size_t k = (i/ipow<size_t>(ncp,d)) % ncp;
                        //wgt *= amat[d][k];
                        wgt *= amat[d][kidx[i][d]];
                        //printf("      iep %ld d %ld k %ld wgt %g\n",i, d, k, wgt);
                    }
                    for (size_t d=0; d<SD; ++d) ep.s[d][iep] += wgt * p.s[d][ip];

                    if (interp_radii) {
                        ep.r[iep] += std::abs(wgt) * p.r[ip];
                        wgtsum[i] += std::abs(wgt);
                    }
                }
            }

            t.epnum[tnode] = numEqps;
        }
    }

    // now adjust particle radii
    if (interp_radii) {
        for (size_t i=0; i<numEqps; ++i) ep.r[iepstart+i] /= wgtsum[i];
    }

    if (dbg) for (size_t i=iepstart; i<iepstop; ++i) printf("    eq part %ld is at %g %g %g mass %g rad %g\n", i, ep.x[0][i], ep.x[1][i], ep.x[2][i], ep.s[0][i], ep.r[i]);

    //printf("  node %d finally has %d equivalent particles, offset %d\n", tnode, t.epnum[tnode], t.epoffset[tnode]);

}
