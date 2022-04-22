/*
 * Parts.h - Set of particles
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#ifdef USE_VC
#include <Vc/Vc>
#endif

#include <cmath>
#include <vector>


#ifdef USE_VC
template <class S> using Vector = std::vector<S, Vc::Allocator<S>>;
#else
template <class S> using Vector = std::vector<S>;
#endif


//
// A set of particles, can be sources or targets
//
// templatized on (S)torage and (A)ccumulator types, and
//   (P)osition (D)imensions, (S)trength (D)ims, (O)utput (D)ims
//
template <class S, class A, int PD, int SD, int OD>
class Parts {
public:
    Parts(const size_t, const bool);
    void resize(const size_t);
    void random_in_cube();
    void smooth_strengths();
    void central_strengths();
    void wave_strengths();
    void zero_vels();
    void reorder_idx(const size_t, const size_t);
    void buffer_end(const size_t);

    // state
    bool are_sources;
    size_t n;
    alignas(32) std::array<Vector<S>, PD> x;

    // actuator (needed by sources)
    alignas(32) std::array<Vector<S>, SD> s;
    alignas(32) Vector<S> r;

    // results (needed by targets)
    alignas(32) std::array<Vector<A>, OD> u;
    alignas(32) std::vector<size_t> gidx;

    // temporary
    alignas(32) std::vector<size_t> lidx;
    alignas(32) std::vector<size_t> itemp;
    alignas(32) Vector<S> ftemp;

    // useful later
    //typename S::value_type state_type;
    //typename A::value_type accumulator_type;
};

template <class S, class A, int PD, int SD, int OD>
Parts<S,A,PD,SD,OD>::Parts(const size_t _num, const bool _aresrcs) {
    are_sources = _aresrcs;
    resize(_num);
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::resize(const size_t _num) {
    n = _num;
    for (int d=0; d<PD; ++d) x[d].resize(n);
    if (are_sources) for (int d=0; d<SD; ++d) s[d].resize(n);
    r.resize(n);
    if (not are_sources) for (int d=0; d<OD; ++d) u[d].resize(n);
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::random_in_cube() {
    for (int d=0; d<PD; ++d) for (auto& _x : x[d]) { _x = -1.0 + 2.0*(S)rand()/(S)RAND_MAX; }
    if (are_sources) for (int d=0; d<SD; ++d) for (auto& _s : s[d]) { _s = (-1.0 + 2.0*(S)rand()/(S)RAND_MAX) / (S)n; }
    for (auto& _r : r) { _r = std::pow((S)n, -1.0/(S)PD); }
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::smooth_strengths() {
    if (not are_sources) return;
    const S factor = 1.0 / (S)n;
    for (int d=0; d<SD; ++d) {
        for (size_t i=0; i<n; i++) {
            s[d][i] = factor * (x[0][i] - x[1][i]);
        }
    }
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::central_strengths() {
    if (not are_sources) return;
    const S factor = 1.0 / (S)n;
    for (size_t i=0; i<n; i++) {
        S dist = 0.0;
        for (int d=0; d<PD; ++d) dist += std::pow(x[d][i]-0.5,2);
        dist = std::sqrt(dist);
        for (int d=0; d<SD; ++d) s[d][i] = factor * std::cos(30.0*std::sqrt(dist)) / (5.0*dist+1.0);
    }
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::wave_strengths() {
    if (not are_sources) return;
    const S factor = 1.0 / (S)n;
    for (size_t i=0; i<n; i++) {
        for (int d=0; d<SD; ++d) s[d][i] = factor * std::cos((d+0.7)*10.0*x[d][i]);
    }
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::zero_vels() {
    if (are_sources) return;
    for (int d=0; d<OD; ++d) for (auto& _u : u[d]) { _u = (S)0.0; }
}

//
// Helper function to reorder the reordering indexes
//
template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::reorder_idx(const size_t pfirst, const size_t plast) {

    // copy the original global index vector gidx into a temporary vector
    std::copy(gidx.begin()+pfirst, gidx.begin()+plast, itemp.begin()+pfirst);

    // scatter values from the temp vector back into the original vector
    for (size_t i=pfirst; i<plast; ++i) gidx[i] = itemp[lidx[i]];
}

// because with Vc we can potentially read more than num entries...
template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::buffer_end(const size_t _veclen) {

    // if we are not using Vc, there's no need to buffer
    if (_veclen == 1) return;
    // if we already have an even multiple of the Vc vector length, also no need
    if (n%_veclen == 0) return;

    // we're here, so let's resize our arrays with sane values
    const size_t bufferedSize = _veclen * (1 + (n-1)/_veclen);
    const size_t lastidx = n-1;

    // now we resize
    for (int d=0; d<PD; ++d) x[d].resize(bufferedSize, x[d][lastidx]);
    if (are_sources) for (int d=0; d<SD; ++d) s[d].resize(bufferedSize, 0.0);
    r.resize(bufferedSize, 1.0);
    if (not are_sources) for (int d=0; d<OD; ++d) u[d].resize(bufferedSize, 0.0);

    // and keep n as-is!
}

