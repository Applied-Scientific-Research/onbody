/*
 * BarycentricLagrange.hpp - functions supporting barycentric Lagrange interpolation
 *
 * Copyright (c) 2022,5-6 Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#include "MathHelper.hpp"
#include "Parts.hpp"
#include "Tree.hpp"

#include <cassert>
#include <cmath>		// for isnan
#include <limits>		// for numeric_limits

#define CLOSE_THRESH 1.e-10

const int32_t maxorder = 20;

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

// indexes to aid in calculation
//template <class S, int PD>
//std::vector<std::array<S,PD>> kidx;


//
// Downward pass: use outputs on source barycentric points to find outputs on arbitrary points in subset target box
//
// sp are source points, always equivalent (Barycentric) points
// tp are target points, can be regular or equivalent points
//

template <class S, class A, int PD, int SD, int OD>
void calcBarycentricDownward(const Parts<S,A,PD,SD,OD>& sp,
                             Parts<S,A,PD,SD,OD>& tp,
                             const int32_t order,
                             const size_t istart, const size_t istop,
                             const size_t iepstart) {

    using Vec = Vc::Vector<S>;
    using Mask = typename Vec::Mask;
    const size_t W = Vec::size();

    // assumes output storage type matches S (true for current instantiations);
    // otherwise instantiate the accumulators below with Vc::Vector<A>

    // set some constants
    const size_t ncp = order+1;                  // number of Chebyshev points per direction
    const size_t numEqps = ipow<size_t>(ncp,PD); // total source barycentric points
    const size_t outer = numEqps / ncp;          // == ipow(ncp, PD-1)
    // number of Chebyshev points padded up to a whole number of vectors
    const size_t ncpP = ((ncp + W - 1) / W) * W;

    // padded copies of the Chebyshev weights and node locations: every vector
    // load below stays in-bounds and unmasked. Padded lanes get weight 0 and a
    // huge location, so they contribute exactly zero and never trip the
    // closeness test. NOTE: node locations are gathered strided (stride=ncp^d)
    // out of sp.x, exactly as in the scalar version.
    const S farAway = std::numeric_limits<S>::max();
    std::vector<S> wkP(ncpP, (S)0);
    std::vector<S> lskP(PD * ncpP);
    for (size_t k=0; k<ncp; ++k) wkP[k] = wk<S>[k];
    for (size_t d=0; d<PD; ++d) {
        const size_t stride = ipow<size_t>(ncp,d);
        for (size_t k=0; k<ncp; ++k) lskP[d*ncpP+k] = sp.x[d][iepstart+stride*k];
        for (size_t k=ncp; k<ncpP; ++k) lskP[d*ncpP+k] = farAway;
    }

    // per-dimension barycentric sub-weights, zero-padded (pads must stay 0!)
    std::array<std::vector<S>,PD> amat;
    for (size_t d=0; d<PD; ++d) amat[d].resize(ncpP, (S)0);

    // contiguous, zero-padded copy of the source outputs. Writing
    // i = j*ncp + k0, the innermost index k0 addresses consecutive source
    // points, so all vector loads in the reduction are unit-stride; the
    // padding guarantees even the longest row ((outer-1)*ncp + ncpP entries)
    // stays inside this buffer -- necessary because the last tree node's
    // block may have no slack past blockSize. The pads MUST be zero: they are
    // multiplied by zero weights, and 0*NaN would otherwise poison the
    // accumulators.
    const size_t bufLen = numEqps - ncp + ncpP;
    std::array<std::vector<S>,OD> ubuf;
    for (size_t d=0; d<OD; ++d) {
        ubuf[d].resize(bufLen, (S)0);
        for (size_t i=0; i<numEqps; ++i) ubuf[d][i] = sp.u[d][iepstart+i];
    }

    // precompute these useful indices - this should be a once-and-done thing,
    // not every tree node (only entries j*ncp for j<outer, d>=1 are consulted)
    std::vector<std::array<size_t,PD>> kidx;
    kidx.resize(numEqps);
    for (size_t d=0; d<PD; ++d) {
        const size_t divisor = ipow<size_t>(ncp,d);
        for (size_t i=0; i<numEqps; ++i) {
            kidx[i][d] = (i/divisor) % ncp;
        }
    }

    const Vec vclose((S)CLOSE_THRESH);

    // loop over target points
    for (size_t ip=istart; ip<istop; ++ip) {

        S denom = (S)1.0;

        // vectorized over coord indices and Cheby points to compute amat
        const S* lp = lskP.data();
        for (size_t d=0; d<PD; ++d, lp += ncpP) {
            const S xd = tp.x[d][ip];

            int32_t flag = -1;
            Vec sumv((S)0);
            for (size_t k=0; k<ncpP; k+=W) {
                const Vec dist = Vec(xd) - Vec(lp+k, Vc::Unaligned);
                const Mask close = Vc::abs(dist) < vclose;
                if (close.isNotEmpty()) {
                    // rare: target sits on a node. Preserve scalar behavior:
                    // the HIGHEST matching k wins.
                    int32_t lastlane = -1;
                    for (size_t l=0; l<W; ++l) if (close[int(l)]) lastlane = int32_t(l);
                    flag = int32_t(k + lastlane);
                }
                Vec a = Vec(wkP.data()+k, Vc::Unaligned) / dist;
                a(close) = (S)0;               // clear singular lanes (incl. inf)
                a.store(&amat[d][k], Vc::Unaligned);
                sumv += a;
            }

            // if a flag was set, remove singularity
            if (flag > -1) {
                // unit weight on the coincident node; that dimension's sum is
                // exactly 1, so it contributes nothing to denom (as in the
                // scalar code)
                for (size_t k=0; k<ncpP; ++k) amat[d][k] = (S)0;
                amat[d][flag] = (S)1;
            } else {
                denom *= sumv.sum();
            }
        }

        // accumulate all numEqps source contributions onto this one target:
        // a vector accumulator per output component, reduced once at the end
        const Vec vdenom((S)1/denom);

        std::array<Vec,OD> accv;
        for (size_t d=0; d<OD; ++d) accv[d] = Vec((S)0);

        // innermost Chebyshev index k0 varies fastest in i and maps to
        // contiguous source memory, so no gathers are needed here
        for (size_t j=0; j<outer; ++j) {
            const size_t jb = j*ncp;
            S wout = (S)1.0;
            for (size_t d=1; d<PD; ++d) wout *= amat[d][kidx[jb][d]]; // hoisted digits
            const Vec vw = vdenom * wout;

            for (size_t k=0; k<ncpP; k+=W) {   // padded tail lanes carry wgt = 0
                const Vec wgt = vw * Vec(&amat[0][k], Vc::Unaligned);
                for (size_t d=0; d<OD; ++d) {
                    accv[d] += wgt * Vec(&ubuf[d][jb+k], Vc::Unaligned);
                }
            }
        }

        for (size_t d=0; d<OD; ++d) tp.u[d][ip] += accv[d].sum();
    }

    // flops
    // (istop-istart)*(1+PD*(1+ncp*5)+numEqps*(PD+2*OD))
}


//
// Loop over all parts in one tree node and calculate the n^d barycentric equivalent parts weights
//
// sp are source points, either equivalent or particles
// tp are target points, always equivalent points
//
template <class S, class A, int PD, int SD, int OD>
void calcBarycentricUpward(const Parts<S,A,PD,SD,OD>& sp,
                           Parts<S,A,PD,SD,OD>& tp,
                           const std::vector<S>& lsk,
                           const std::vector<std::array<size_t,PD>>& kidx,
                           std::vector<S>& wgtsum,
                           const size_t ncp, const size_t numEqps,
                           const size_t istart, const size_t istop,
                           const size_t iepstart, const bool interp_radii) {

    using Vec = Vc::Vector<S>;
    using Mask = typename Vec::Mask;
    const size_t W = Vec::size();

    // number of Chebyshev points padded to a whole number of vectors
    const size_t ncpP = ((ncp + W - 1) / W) * W;

    // padded copies of the Chebyshev nodes/weights: every vector load below stays
    // in-bounds and unmasked. Padded lanes get weight 0 and a huge location, so they
    // contribute exactly zero and never trip the closeness test.
    const S farAway = std::numeric_limits<S>::max();
    std::vector<S> wkP(ncpP, (S)0);
    std::vector<S> lskP(PD * ncpP);
    for (size_t k=0; k<ncp; ++k) wkP[k] = wk<S>[k];
    for (size_t d=0; d<PD; ++d) {
        for (size_t k=0; k<ncp; ++k) lskP[d*ncpP+k] = lsk[d*ncp+k];
        for (size_t k=ncp; k<ncpP; ++k) lskP[d*ncpP+k] = farAway;
    }

    // per-dimension barycentric sub-weights, zero-padded (pads must stay 0!)
    std::array<std::vector<S>,PD> amat;
    for (size_t d=0; d<PD; ++d) amat[d].resize(ncpP, (S)0);

    // output accumulators, flushed into tp once at the end of this call.
    // i = j*ncp + k0: longest row touched is (outer-1)*ncp + ncpP entries.
    const size_t outer = numEqps / ncp;             // == ipow(ncp, PD-1)
    const size_t accLen = numEqps - ncp + ncpP;
    std::array<std::vector<S>,SD> uacc;
    for (size_t d=0; d<SD; ++d) uacc[d].resize(accLen, (S)0);
    std::vector<S> uaccR, uaccW;
    if (interp_radii) { uaccR.resize(accLen, (S)0); uaccW.resize(accLen, (S)0); }

    const Vec vclose((S)CLOSE_THRESH);

    for (size_t ip=istart; ip<istop; ++ip) {
        S denom = (S)1.0;

        const S* lp = lskP.data();
        for (size_t d=0; d<PD; ++d, lp += ncpP) {
            const S xd = sp.x[d][ip];

            int32_t flag = -1;
            Vec sumv((S)0);
            for (size_t k=0; k<ncpP; k+=W) {
                const Vec dist = Vec(xd) - Vec(lp+k, Vc::Unaligned);
                const Mask close = Vc::abs(dist) < vclose;
                if (close.isNotEmpty()) {
                    // rare: particle sits on a node. Preserve scalar behavior:
                    // the HIGHEST matching k wins.
                    int32_t lastlane = -1;
                    for (size_t l=0; l<W; ++l) if (close[int(l)]) lastlane = int32_t(l);
                    flag = int32_t(k + lastlane);
                }
                Vec a = Vec(wkP.data()+k, Vc::Unaligned) / dist;
                a(close) = (S)0;                    // clear singular lanes (incl. inf)
                a.store(&amat[d][k], Vc::Unaligned);
                sumv += a;
            }

            if (flag > -1) {
                // remove the singularity: unit weight on the coincident node
                for (size_t k=0; k<ncpP; ++k) amat[d][k] = (S)0;
                amat[d][flag] = (S)1;
                // dimension sum is exactly 1 -> contributes nothing to denom
            } else {
                denom *= sumv.sum();
            }
        }
        const Vec vdenom((S)1/denom);

        // innermost Chebyshev index k0 varies fastest in i and maps to contiguous
        // equivalent-particle memory, so no gathers are needed here
        for (size_t j=0; j<outer; ++j) {
            const size_t jb = j*ncp;
            S wout = (S)1.0;
            for (size_t d=1; d<PD; ++d) wout *= amat[d][kidx[jb][d]];   // hoisted digits
            const Vec vw = vdenom * wout;

            for (size_t k=0; k<ncpP; k+=W) {        // padded tail lanes carry wgt = 0
                const Vec wgt = vw * Vec(&amat[0][k], Vc::Unaligned);
                for (size_t d=0; d<SD; ++d) {
                    Vec acc = Vec(&uacc[d][jb+k], Vc::Unaligned) + wgt * sp.s[d][ip];
                    acc.store(&uacc[d][jb+k], Vc::Unaligned);
                }
                if (interp_radii) {
                    const Vec aw = Vc::abs(wgt);
                    Vec r = Vec(&uaccR[jb+k], Vc::Unaligned) + aw * sp.r[ip];
                    r.store(&uaccR[jb+k], Vc::Unaligned);
                    Vec w = Vec(&uaccW[jb+k], Vc::Unaligned) + aw;
                    w.store(&uaccW[jb+k], Vc::Unaligned);
                }
            }
        }
    }

    // flush accumulated outputs into the target (equivalent) particles
    for (size_t d=0; d<SD; ++d) {
        size_t i = 0;
        for (; i+W <= numEqps; i += W) {
            const Vec acc = Vec(&tp.s[d][iepstart+i], Vc::Unaligned)
                          + Vec(&uacc[d][i],         Vc::Unaligned);
            acc.store(&tp.s[d][iepstart+i], Vc::Unaligned);
        }
        for (; i<numEqps; ++i) tp.s[d][iepstart+i] += uacc[d][i];
    }
    if (interp_radii) {
        size_t i = 0;
        for (; i+W <= numEqps; i += W) {
            const Vec r = Vec(&tp.r[iepstart+i], Vc::Unaligned) + Vec(&uaccR[i], Vc::Unaligned);
            r.store(&tp.r[iepstart+i], Vc::Unaligned);
            const Vec w = Vec(&wgtsum[i], Vc::Unaligned) + Vec(&uaccW[i], Vc::Unaligned);
            w.store(&wgtsum[i], Vc::Unaligned);
        }
        for (; i<numEqps; ++i) { tp.r[iepstart+i] += uaccR[i]; wgtsum[i] += uaccW[i]; }
    }

    // flops: (istop-istart)*(1+PD*(1+ncp*5)+numEqps*(PD+2*OD))
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
    const bool interp_radii = false;

    // this this a leaf node?
    if (t.num[tnode] <= p.blockSize) return;
    if (dbg) printf("  node %ld has %ld particles\n", tnode, t.num[tnode]);

    // set the locations and weights of the barycentric particles
    // we are always in an "omp single" construct when we reach this
    if (tnode == 1) {
        (void) set_sk<S>(order);
        (void) set_wk<S>(order);
    }

    // make sure the children are done before proceeding
    #pragma omp task shared(p,ep,t)
    (void) calcBarycentricLagrange(p, ep, t, order, 2*tnode);
    #pragma omp task shared(p,ep,t)
    (void) calcBarycentricLagrange(p, ep, t, order, 2*tnode+1);
    #pragma omp taskwait

    // set some constants
    const size_t ncp = order+1;		// number of Chebyshev points in each direction
    const size_t numEqps = ipow<size_t>(ncp,PD);
    //printf("    ncp %ld and numEqps %ld\n", ncp, numEqps);
    assert(numEqps <= ep.blockSize && "ERROR (calcBarycentricLagrange): requested order too large for blockSize");

    // continue using blockSize for offset
    t.epoffset[tnode] = tnode * ep.blockSize;
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
                *lsk_iter = t.nc[d][tnode] + (S)0.5 * sk<S>[k] * t.ns[d][tnode];
                ++lsk_iter;
            }
        }
    }

    // precompute these useful indices - this should be a once-and-done thing, not every tree node
    std::vector<std::array<size_t,PD>> kidx;
    kidx.resize(numEqps);
    for (size_t d=0; d<PD; ++d) {
        const size_t divisor = ipow<size_t>(ncp,d);
        for (size_t i=0; i<numEqps; ++i) {
            kidx[i][d] = (i/divisor) % ncp;
        }
    }
    // really, why can't we just keep track of (d*i) and increment from 0 to ncp-1 ?

    // note that t.x[d][tnode] is the center of mass - not the center of the cluster!!!
    // the cluster size is t.ns[d][tnode]
    // geometric center is t.nc[d][tnode]
    for (size_t d=0; d<PD; ++d) {
        //const size_t divisor = ipow<size_t>(ncp,d);
        //printf("    d %ld and divisor %ld\n", d, divisor);
        for (size_t i=0; i<numEqps; ++i) {
            //ep.x[d][iepstart+i] = t.nc[d][tnode] + (S)0.5 * sk<S>[(i/divisor)%ncp] * t.ns[d][tnode];
            ep.x[d][iepstart+i] = t.nc[d][tnode] + (S)0.5 * sk<S>[kidx[i][d]] * t.ns[d][tnode];
        }
    }

    // and locate the remainder of the particles (unused) to the cell center
    for (size_t i=iepstop; i<iepstart+ep.blockSize; ++i) {
        for (size_t d=0; d<PD; ++d) ep.x[d][i] = t.nc[d][tnode];
    }

    if (dbg) for (size_t i=iepstart; i<iepstop; ++i) printf("    eq part %ld is at %g %g %g\n", i, ep.x[0][i], ep.x[1][i], ep.x[2][i]);
    //for (size_t i=iepstart; i<iepstart+ep.blockSize; ++i) printf("    eq part %ld is at %g %g %g\n", i, ep.x[0][i], ep.x[1][i], ep.x[2][i]);

    // set all equiv. particles weights
    if (ep.are_sources) {
        for (size_t i=iepstart; i<iepstart+ep.blockSize; ++i) {
            for (size_t d=0; d<SD; ++d) ep.s[d][i] = 0.0;
        }
    }

    // initialize radii to zero, or just copy the first particle radius
    if (interp_radii) {
        for (size_t i=iepstart; i<iepstart+ep.blockSize; ++i) ep.r[i] = 0.0;
    } else {
        for (size_t i=iepstart; i<iepstart+ep.blockSize; ++i) ep.r[i] = p.r[t.ioffset[tnode]];
    }

    // store a sum of weights to be used for particle radii
    std::vector<S> wgtsum(ep.blockSize, 0.0);

    // loop over children, adding equivalent particles to our list
    for (size_t ichild = 2*tnode; ichild < 2*tnode+2; ++ichild) {
        if (dbg) printf("  child %ld has %ld particles\n", ichild, t.num[ichild]);

        // split on whether this child is a leaf node or not
        if (t.num[ichild] > p.blockSize) {

            // not a leaf node
            // now we read those equivalent particles and make higher-level equivalents

            // here istart and istop are the previous equivalent particles
            const size_t istart = t.epoffset[ichild];
            const size_t istop = istart + t.epnum[ichild];
            if (dbg) printf("    starting with equiv particles %ld to %ld\n", istart, istop);

            // and here is the range of new equivalent particles
            if (dbg) printf("    putting into %ld new equiv particles %ld to %ld\n", numEqps, iepstart, iepstop);

            // now do the work
            if (p.are_sources and ep.are_sources) {
                calcBarycentricUpward<S,A,PD,SD,OD>(ep, ep, lsk, kidx, wgtsum, ncp, numEqps, istart, istop, iepstart, interp_radii);
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

            // now do the work - but only if strengths exist
            if (p.are_sources and ep.are_sources) {
                calcBarycentricUpward<S,A,PD,SD,OD>(p, ep, lsk, kidx, wgtsum, ncp, numEqps, istart, istop, iepstart, interp_radii);
            }

            t.epnum[tnode] = numEqps;
        }
    }

    // now adjust particle radii
    if (interp_radii) {
        for (size_t i=0; i<numEqps; ++i) ep.r[iepstart+i] /= wgtsum[i];
    }

    if (dbg) for (size_t i=iepstart; i<iepstop; ++i) printf("    eq part %ld is at %g %g %g mass %g rad %g\n", i, ep.x[0][i], ep.x[1][i], ep.x[2][i], ep.s[0][i], ep.r[i]);

    //printf("  node %ld finally has %ld equivalent particles, offset %ld\n", tnode, t.epnum[tnode], t.epoffset[tnode]);

}
