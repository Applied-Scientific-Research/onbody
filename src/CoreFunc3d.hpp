/*
 * CoreFunc3d.h - core functions useful for particle methods
 *
 * Copyright (c) 2017-20, Mark J Stock <markjstock@gmail.com>
 *
 * Look in ~/asr/version3/Omega3D/src/CoreFunc.h for more
 */

#pragma once

#define USE_RM_KERNEL
//#define USE_EXPONENTIAL_KERNEL

#include "MathHelper.hpp"

#ifdef USE_VC
#include <Vc/Vc>
#endif

#include <cmath>


#ifdef USE_RM_KERNEL
//
// core functions - Rosenhead-Moore
//
template <class S>
static inline S core_func (const S distsq, const S sr) {
  const S r2 = distsq + sr*sr;
  return oor1p5(r2);
}
static inline int flops_tp_nograds () { return 5; }

template <class S>
static inline void core_func (const S distsq, const S sr,
                              S* const __restrict__ r3, S* const __restrict__ bbb) {
  const S r2 = distsq + sr*sr;
  *r3 = oor1p5(r2);
  *bbb = S(-3.0) * (*r3) * my_recip(r2);
}
static inline int flops_tp_grads () { return 8; }
#endif

#ifdef USE_EXPONENTIAL_KERNEL
//
// core functions - compact exponential
//
#ifdef USE_VC

template <class S>
static inline S core_func (const S distsq, const S sr) {
  const S dist = Vc::sqrt(distsq);
  const S corefac = Vc::reciprocal(sr*sr*sr);
  const S ood3 = Vc::reciprocal(distsq * dist);
  const S reld3 = corefac / ood3;
  // 7 flops to here
  S returnval = ood3;
  returnval(reld3 < S(16.0)) = ood3 * (S(1.0) - Vc::exp(-reld3));
  returnval(reld3 < S(0.001)) = corefac;
  return returnval;
}

template <class S>
static inline void core_func (const S distsq, const S sr,
                              S* const __restrict__ r3, S* const __restrict__ bbb) {
  const S dm1 = Vc::rsqrt(distsq);
  const S corefac = Vc::reciprocal(sr*sr*sr);
  const S d3 = distsq * distsq * dm1;
  const S reld3 = d3 * corefac;
  const S dm3 = Vc::reciprocal(d3);
  const S dm2 = dm1 * dm1;
  // 9 flops to here

  S myr3 = dm3;
  S mybbb = S(-3.0) * dm3 * dm2;
  const S expreld3 = Vc::exp(-reld3);
  // what is auto? Vc::float_m, Vc::Mask<float> usually
  const auto mid = (reld3 < 16.f);
  myr3(mid) = (S(1.0) - expreld3) * dm3;
  mybbb(mid) = S(3.0) * (corefac*expreld3 - myr3) * dm2;
  const auto close = (reld3 < 0.001f);
  myr3(close) = corefac;
  mybbb(close) = S(-1.5) * dm2 * reld3 * corefac;
  // probably 11 more flops
  *r3 = myr3;
  *bbb = mybbb;
}

template <>
inline void core_func (const float distsq, const float sr,
                       float* const __restrict__ r3, float* const __restrict__ bbb) {
  const float dist = std::sqrt(distsq);
  const float corefac = 1.0f / std::pow(sr,3);
  const float d3 = distsq * dist;
  const float reld3 = d3 * corefac;
  const float dm3 = 1.0f / d3;
  const float dm2 = 1.0f / distsq;

  if (reld3 > 16.0f) {
    *r3 = dm3;
    *bbb = -3.0f * dm3 * dm2;
  } else if (reld3 < 0.001f) {
    *r3 = corefac;
    *bbb = -1.5f * dist * corefac * corefac;
  } else {
    const float expreld3 = std::exp(-reld3);
    *r3 = (1.0f - expreld3) * dm3;
    *bbb = 3.0f * (corefac*expreld3 - *r3) * dm2;
  }
}

template <>
inline void core_func (const double distsq, const double sr,
                       double* const __restrict__ r3, double* const __restrict__ bbb) {
  const double dist = std::sqrt(distsq);
  const double corefac = 1.0 / std::pow(sr,3);
  const double d3 = distsq * dist;
  const double reld3 = d3 * corefac;
  const double dm3 = 1.0 / d3;
  const double dm2 = 1.0 / distsq;

  if (reld3 > 16.0) {
    *r3 = dm3;
    *bbb = -3.0 * dm3 * dm2;
  } else if (reld3 < 0.001) {
    *r3 = corefac;
    *bbb = -1.5 * dist * corefac * corefac;
  } else {
    const double expreld3 = std::exp(-reld3);
    *r3 = (1.0 - expreld3) * dm3;
    *bbb = 3.0 * (corefac*expreld3 - *r3) * dm2;
  }
}
#else

template <class S>
static inline S core_func (const S distsq, const S sr) {
  const S dist = std::sqrt(distsq);
  const S corefac = S(1.0) / std::pow(sr,3);
  const S ood3 = S(1.0) / (distsq * dist);
  const S reld3 = corefac / ood3;

  if (reld3 > S(16.0)) {
    return ood3;
  } else if (reld3 < S(0.001)) {
    return corefac;
  } else {
    return ood3 * (S(1.0) - std::exp(-reld3));
  }
}

template <class S>
static inline void core_func (const S distsq, const S sr,
                              S* const __restrict__ r3, S* const __restrict__ bbb) {
  const S dist = std::sqrt(distsq);
  const S corefac = S(1.0) / std::pow(sr,3);
  const S d3 = distsq * dist;
  const S reld3 = d3 * corefac;
  const S dm3 = S(1.0) / d3;
  const S dm2 = S(1.0) / distsq;
  // 8 flops to here

  if (reld3 > S(16.0)) {
    *r3 = dm3;
    *bbb = S(-3.0) * dm3 * dm2;
    // this is 3 flops and is very likely
  } else if (reld3 < S(0.001)) {
    *r3 = corefac;
    *bbb = S(-1.5) * dist * corefac * corefac;
    // this is 4 flops
  } else {
    const S expreld3 = std::exp(-reld3);
    *r3 = (S(1.0) - expreld3) * dm3;
    *bbb = S(3.0) * (corefac*expreld3 - *r3) * dm2;
    // this is 7 flops and also very likely
  }
}
#endif

static inline int flops_tp_nograds () { return 9; }
static inline int flops_tp_grads () { return 15; }
#endif


#ifdef USE_V2_KERNEL
//
// core functions - Vatistas n=2
//
template <class S>
static inline S core_func (const S distsq, const S sr) {
  const S s2 = sr*sr;
  const S denom = distsq*distsq + s2*s2;
  const S rsqd = my_rsqrt(denom);
  return rsqd*my_sqrt(rsqd);
}

static inline int flops_tp_nograds () { return 7; }
#endif

