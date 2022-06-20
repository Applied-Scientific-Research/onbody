/*
 * CoreFunc2d.h - core functions useful for particle methods
 *
 * Copyright (c) 2017-22, Mark J Stock <markjstock@gmail.com>
 *
 * Look in ~/asr/version3/Omega2D/src/CoreFunc.h for more
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
template <class S>
static inline S core_func (const S distsq, const S sr) {
  const S r2 = distsq + sr*sr;
  return my_recip(r2);
}
static inline int flops_tp_nograds () { return 3; }

template <class S>
static inline S core_func (const S distsq, const S sr, const S tr) {
  const S r2 = distsq + sr*sr + tr*tr;
  return my_recip(r2);
}
static inline int flops_tpr_nograds () { return 5; }

#endif

#ifdef USE_EXPONENTIAL_KERNEL
#ifdef USE_VC
template <class S>
static inline S exp_cond (const S ood2, const S corefac, const S reld2) {
  S returnval = ood2;
  returnval(reld2 < S(16.0)) = ood2 * (S(1.0) - Vc::exp(-reld2));
  returnval(reld2 < S(0.001)) = corefac;
  return returnval;
}
template <>
inline float exp_cond (const float ood2, const float corefac, const float reld2) {
  if (reld2 > 16.0f) {
    return ood2;
  } else if (reld2 < 0.001f) {
    return corefac;
  } else {
    return ood2 * (1.0f - std::exp(-reld2));
  }
}
template <>
inline double exp_cond (const double ood2, const double corefac, const double reld2) {
  if (reld2 > 16.0) {
    return ood2;
  } else if (reld2 < 0.001) {
    return corefac;
  } else {
    return ood2 * (1.0 - std::exp(-reld2));
  }
}
#else
template <class S>
static inline S exp_cond (const S ood2, const S corefac, const S reld2) {
  if (reld2 > 16.0f) {
    return ood2;
    // 1 flop (comparison)
  } else if (reld2 < 0.001f) {
    return corefac;
    // 2 flops
  } else {
    return ood2 * (1.0f - std::exp(-reld2));
    // 3 flops
  }
}
#endif

template <class S>
static inline S core_func (const S distsq, const S sr) {
  const S ood2 = my_recip(distsq+S(1.e-6));
  const S corefac = my_recip(sr*sr);
  const S reld2 = corefac / ood2;
  // 4 flops to here
  return exp_cond(ood2, corefac, reld2);
}
static inline int flops_tp_nograds () { return 9; }

template <class S>
static inline S core_func (const S distsq, const S sr, const S tr) {
  const S ood2 = my_recip(distsq+S(1.e-6));
  const S corefac = my_recip(sr*sr + tr*tr);
  const S reld2 = corefac / ood2;
  // 4 flops to here
  return exp_cond(ood2, corefac, reld2);
}
static inline int flops_tpr_nograds () { return 11; }
#endif

