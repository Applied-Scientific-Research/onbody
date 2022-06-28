/*
 * Tree.h - simple tree for onbody
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 */

#pragma once

#ifdef USE_VC
#include <Vc/Vc>
#endif

#include <cstdint>
#include <stdio.h>
#include <cmath>
#include <vector>

#define PARTIAL_SORT

#ifdef USE_VC
template <class S> using Vector = std::vector<S, Vc::Allocator<S>>;
#else
template <class S> using Vector = std::vector<S>;
#endif

// the basic unit of direct sum work is blockSize by blockSize
const size_t blockSize = 128;

//
// Find index of msb of uint32
// from http://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
//
static inline uint32_t log_2(const uint32_t x) {
    if (x == 0) return 0;
    return (31 - __builtin_clz (x));
}


//
// A tree, made of a structure of arrays
//
// 0 is empty, root node is 1, children are 2,3, their children 4,5 and 6,7
// arrays always have 2^levels boxes allocated, even if some are not used
// this way, node i children are 2*i and 2*i+1
//
// templatized on (S)torage type, (P)osition (D)imensions, and (S)trength (D)ims
//
template <class S, int PD, int SD>
class Tree {
public:
    Tree(size_t);
    void resize(size_t);
    void print(size_t);

    // number of levels in the tree
    int levels;
    // number of nodes in the tree (always 2^l)
    int numnodes;

    // tree node centers of mass
    alignas(32) std::array<Vector<S>, PD> x;
    // node geometric center
    alignas(32) std::array<Vector<S>, PD> nc;
    // node size
    alignas(32) std::array<Vector<S>, PD> ns;
    // node radius
    alignas(32) Vector<S> nr;
    // node particle radius
    alignas(32) Vector<S> pr;
    // node strengths
    alignas(32) std::array<Vector<S>, SD> s;

    // real point offset and count
    alignas(32) std::vector<size_t> ioffset;		// is this redundant?
    alignas(32) std::vector<size_t> num;
    // equivalent point offset and count
    alignas(32) std::vector<size_t> epoffset;		// is this redundant?
    alignas(32) std::vector<size_t> epnum;
};

template <class S, int PD, int SD>
Tree<S,PD,SD>::Tree(const size_t _num) {
    if (_num==0) return;
    // _num is number of elements this tree needs to store
    uint32_t numLeaf = 1 + ((_num-1)/blockSize);
    //printf("  %d nodes at leaf level\n", numLeaf);
    levels = 1 + log_2(2*numLeaf-1);
    //printf("  makes %d levels in tree\n", levels);
    numnodes = 1 << levels;
    //printf("  and %d total nodes in tree\n", numnodes);
    resize(numnodes);
}

template <class S, int PD, int SD>
void Tree<S,PD,SD>::resize(const size_t _num) {
    numnodes = _num;
    for (int d=0; d<PD; ++d) x[d].resize(numnodes);
    for (int d=0; d<PD; ++d) nc[d].resize(numnodes);
    for (int d=0; d<PD; ++d) ns[d].resize(numnodes);
    nr.resize(numnodes);
    pr.resize(numnodes);
    for (int d=0; d<SD; ++d) s[d].resize(numnodes);
    ioffset.resize(numnodes);
    num.resize(numnodes);
    std::fill(num.begin(), num.end(), 0);
    epoffset.resize(numnodes);
    epnum.resize(numnodes);
}

template <class S, int PD, int SD>
void Tree<S,PD,SD>::print(const size_t _num) {
    printf("\n%dD tree with %d levels\n", PD, levels);
    for (size_t i=1; i<numnodes && i<_num; ++i) {
        printf("  %ld  %ld %ld  %g\n",i, num[i], ioffset[i], s[i]);
    }
}

