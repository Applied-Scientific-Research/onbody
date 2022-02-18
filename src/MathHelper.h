/*
 * MathHelper.h - Non-class helper inlines for core functions, kernels, influences
 *
 * (c)2020 Applied Scientific Research, Inc.
 *         Mark J Stock <markjstock@gmail.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#ifdef _WIN32
#define __restrict__ __restrict
#endif

#ifdef USE_VC
#include <Vc/Vc>
#endif

#include <cmath>

// helper functions: cast, exp, recip, sqrt, rsqrt, rcbrt, and others

// casting from S to A
template <class S, class A>
#ifdef USE_VC
static inline A mycast (const S _in) { return Vc::simd_cast<A>(_in); }
#else
static inline A mycast (const S _in) { return _in; }
#endif
// specialize, in case the non-vectorized version of nbody_kernel is called
template <> inline float mycast (const float _in) { return _in; }
template <> inline double mycast (const float _in) { return (double)_in; }
template <> inline double mycast (const double _in) { return _in; }

#ifdef USE_VC
template <class S>
static inline S my_exp(const S _in) {
  return Vc::exp(_in);
}
template <>
inline float my_exp(const float _in) {
  return std::exp(_in);
}
template <>
inline double my_exp(const double _in) {
  return std::exp(_in);
}
#else
template <class S>
static inline S my_exp(const S _in) {
  return std::exp(_in);
}
#endif

#ifdef USE_VC
template <class S>
static inline S my_recip(const S _in) {
  return Vc::reciprocal(_in);
}
template <>
inline float my_recip(const float _in) {
  return 1.0f / _in;
}
template <>
inline double my_recip(const double _in) {
  return 1.0 / _in;
}
#else
template <class S>
static inline S my_recip(const S _in) {
  return S(1.0) / _in;
}
#endif

#ifdef USE_VC
template <class S>
static inline S my_sqrt(const S _in) {
  return Vc::sqrt(_in);
}
template <>
inline float my_sqrt(const float _in) {
  return std::sqrt(_in);
}
template <>
inline double my_sqrt(const double _in) {
  return std::sqrt(_in);
}
#else
template <class S>
static inline S my_sqrt(const S _in) {
  return std::sqrt(_in);
}
#endif

#ifdef USE_VC
template <class S>
static inline S my_rsqrt(const S _in) {
  return Vc::rsqrt(_in);
}
template <>
inline float my_rsqrt(const float _in) {
  return 1.0f / std::sqrt(_in);
}
template <>
inline double my_rsqrt(const double _in) {
  return 1.0 / std::sqrt(_in);
}
#else
template <class S>
static inline S my_rsqrt(const S _in) {
  return S(1.0) / std::sqrt(_in);
}
#endif

#ifdef USE_VC
template <class S>
static inline S my_rcbrt(const S _in) {
  return Vc::exp(S(-0.3333333)*Vc::log(_in));
}
template <>
inline float my_rcbrt(const float _in) {
  return 1.0f / std::cbrt(_in);
}
template <>
inline double my_rcbrt(const double _in) {
  return 1.0 / std::cbrt(_in);
}
#else
template <class S>
static inline S my_rcbrt(const S _in) {
  return S(1.0) / std::cbrt(_in);
}
#endif

#ifdef USE_VC
template <class S>
static inline S my_halflog (const S _x) {
  return S(0.5f) * Vc::log(_x);
}
template <> inline float my_halflog (const float _x) {
  return 0.5f * std::log(_x);
}
template <> inline double my_halflog (const double _x) {
  return 0.5 * std::log(_x);
}
#else
template <class S>
static inline S my_halflog (const S _x) {
  return 0.5f * std::log(_x);
}
#endif

#ifdef USE_VC
template <class S>
static inline S oor2p5(const S _in) {
  //return Vc::reciprocal(_in*_in*Vc::sqrt(_in));	// 234 GFlop/s
  return Vc::rsqrt(_in) * Vc::reciprocal(_in*_in);	// 269 GFlop/s
}
template <>
inline float oor2p5(const float _in) {
  return 1.0f / (_in*_in*std::sqrt(_in));
}
template <>
inline double oor2p5(const double _in) {
  return 1.0 / (_in*_in*std::sqrt(_in));
}
#else
template <class S>
static inline S oor2p5(const S _in) {
  return S(1.0) / (_in*_in*std::sqrt(_in));
}
#endif

#ifdef USE_VC
template <class S>
static inline S oor1p5(const S _in) {
  //return Vc::reciprocal(_in*Vc::sqrt(_in));		// 243 GFlop/s
  return Vc::rsqrt(_in) * Vc::reciprocal(_in);		// 302 GFlop/s
}
template <>
inline float oor1p5(const float _in) {
  return 1.0f / (_in*std::sqrt(_in));
}
template <>
inline double oor1p5(const double _in) {
  return 1.0 / (_in*std::sqrt(_in));
}
#else
template <class S>
static inline S oor1p5(const S _in) {
  return S(1.0) / (_in*std::sqrt(_in));
}
#endif

#ifdef USE_VC
template <class S>
static inline S oor0p75(const S _in) {
  const S rsqd = Vc::rsqrt(_in);
  //return rsqd*Vc::sqrt(rsqd);				// 265 GFlop/s
  return rsqd*rsqd*Vc::rsqrt(rsqd);			// 301 GFlop/s
}
template <>
inline float oor0p75(const float _in) {
  const float sqd = std::sqrt(_in);
  return 1.0f / (sqd*std::sqrt(sqd));
}
template <>
inline double oor0p75(const double _in) {
  const double sqd = std::sqrt(_in);
  return 1.0 / (sqd*std::sqrt(sqd));
}
#else
template <class S>
static inline S oor0p75(const S _in) {
  const S sqd = std::sqrt(_in);
  return S(1.0) / (sqd*std::sqrt(sqd));
}
#endif

