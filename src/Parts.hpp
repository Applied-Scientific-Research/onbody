/*
 * Parts.h - Set of particles
 *
 * Copyright (c) 2017-22, Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#ifdef USE_VC
#include <Vc/Vc>
#endif

#include <cmath>
#include <vector>
#include <array>
#include <random>


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
    Parts(const size_t _num, const bool _aresrcs, const size_t _blocksize=128);
    void resize(const size_t);
    void random_in_cube();
    void random_in_cube(std::mt19937);
    void random_in_disk(std::mt19937);
    void smooth_strengths();
    void randomize_radii();
    void randomize_radii(std::mt19937);
    void central_strengths();
    void wave_strengths();
    void zero_vels();
    void reorder_idx(const size_t, const size_t);
    void buffer_end(const size_t);

    // state
    bool are_sources;
    size_t n;
    // the basic unit of direct sum work
    size_t blockSize;

    // all points
    alignas(32) std::array<Vector<S>, PD> x;
    alignas(32) Vector<S> r;

    // actuator (needed by sources)
    alignas(32) std::array<Vector<S>, SD> s;

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
Parts<S,A,PD,SD,OD>::Parts(const size_t _num, const bool _aresrcs, const size_t _blocksize) {
    are_sources = _aresrcs;
    blockSize = _blocksize;
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
void Parts<S,A,PD,SD,OD>::random_in_cube(std::mt19937 _engine) {
    std::uniform_real_distribution<S> zmean_dist(-1.0, 1.0);
    for (int d=0; d<PD; ++d) for (auto& _x : x[d]) { _x = zmean_dist(_engine); }
    if (are_sources) {
        const S factor = 1.0 / (S)n;
        for (int d=0; d<SD; ++d) for (auto& _s : s[d]) { _s = zmean_dist(_engine) * factor; }
    }
    const S thisrad = std::pow((S)n, -1.0/(S)PD);
    for (auto& _r : r) { _r = thisrad; }
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::random_in_disk(std::mt19937 _engine) {
    // first one is the central star
    x[0][0] = 0.0; x[1][0] = 0.0; x[2][0] = 0.0; s[0][0] = 1.0;

    // subsequent ones are low-mass orbiters
    std::uniform_real_distribution<S> theta_dist(0.0, 2.0*3.14159265358979);
    for (size_t i=1; i<n; i++) {
        const S rad = 0.1 + 5.0*(S)i/(S)n;
        const S theta = theta_dist(_engine);
        x[0][i] = rad * std::cos(theta);
        x[1][i] = rad * std::sin(theta);
        for (int d=2; d<PD; ++d) x[d][i] = 0.0;
    }
    if (are_sources) {
        const S mass = 0.1 / (S)n;
        for (auto& _s : s[0]) { _s = mass; }
        for (int d=1; d<SD; ++d) for (auto& _s : s[d]) { _s = 0.0; }
        s[0][0] = 1.0;
    }
    const S thisrad = std::pow(0.1/(S)n, 2);
    for (auto& _r : r) { _r = thisrad; }
    r[0] = 0.00465;
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
void Parts<S,A,PD,SD,OD>::randomize_radii() {
    for (auto& _r : r) { _r *= 0.5 + ((S)rand()/(S)RAND_MAX); }
}

template <class S, class A, int PD, int SD, int OD>
void Parts<S,A,PD,SD,OD>::randomize_radii(std::mt19937 _engine) {
    std::uniform_real_distribution<S> zmean_dist(0.5, 1.5);
    for (auto& _r : r) { _r *= zmean_dist(_engine); }
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

