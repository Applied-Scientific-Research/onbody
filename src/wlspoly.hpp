//
// A templated class for weighted least squares polynomials
//
// templatized on data type, number of spatial dimensions, and polynomial order
//
// Portions of this code are
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
// Written by Jim Leek <leek2@llnl.gov> and Adrian Wong <adrianskw@gmail.com>
// LLNL-CODE-704097
// All rights reserved.
// This file is part of C++ PolyFit.·
// For details, see https://github.com/llnl/CxxPolyFit.
// Please also CxxPolyFit/LICENSE.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//

#include <vector>
#include <Eigen/Dense>

//
// First, helped constexpr to create numBasis and powers at compile-time
//
constexpr int32_t getNumBasis(const int32_t ndim, const int32_t norder) {
    int32_t outNumBasis = 0;

    switch(ndim) {

    case 1:
        return norder+1;

    case 2:
        for (int32_t ii = 0; ii < norder+1; ++ii) {
            for (int32_t j1 = norder; j1 >= 0; --j1) {
                for (int32_t j2 = norder; j2 >= 0; --j2) {
                    if (j1+j2==ii) ++outNumBasis;
                }
            }
        }
        return outNumBasis;

    case 3:
        for (int32_t ii = 0; ii < norder+1; ++ii) {
            for (int32_t j1 = norder; j1 >= 0; --j1) {
                for (int32_t j2 = norder; j2 >= 0; --j2) {
                    for (int32_t j3 = norder; j3 >= 0; --j3) {
                        if (j1+j2+j3==ii) ++outNumBasis;
                    }
                }
            }
        }
        return outNumBasis;

    default:
        return 0;
    }

    return 0;
}

template <int32_t NPOWERS>
using VectorBasis = Eigen::Matrix<int32_t, NPOWERS, 1>;

template <int32_t NDIM, int32_t NORDER>
auto getBasisPowers() {

    constexpr int32_t numBasis = getNumBasis(NDIM, NORDER);
    VectorBasis<numBasis*NDIM> powers;
    int32_t basisNum = 0;

    switch(NDIM) {

    case 1:
        for (int32_t ii = 0; ii < NORDER+1; ++ii) {
            powers[ii] = ii;
        }
        break;

    case 2:
        for (int32_t ii = 0; ii < NORDER+1; ++ii) {
            for (int32_t j1 = NORDER; j1 >= 0; --j1) {
                for (int32_t j2 = NORDER; j2 >= 0; --j2) {
                    if (j1+j2==ii) {
                        powers[basisNum] = j1;				//1st dimension
                        powers[numBasis + basisNum] = j2;	//2nd dimension
                        ++basisNum;
                    }
                }
            }
        }
        break;

    case 3:
        for (int32_t ii = 0; ii < NORDER+1; ++ii) {
            for (int32_t j1 = NORDER; j1 >= 0; --j1) {
                for (int32_t j2 = NORDER; j2 >= 0; --j2) {
                    for (int32_t j3 = NORDER; j3 >= 0; --j3) {
                        if (j1+j2+j3==ii) {
                            powers[basisNum] = j1;				//1st dimension
                            powers[numBasis + basisNum] = j2;	//2nd dimension
                            powers[2*numBasis + basisNum] = j3;	//3rd dimension
                            ++basisNum;
                        }
                    }
                }
            }
        }
        break;
    }

    return powers;
}


template <class T, int32_t NDIM, int32_t NORDER>
class WLSPoly {
public:

    // ctor computes some properties and readies the arrays
    WLSPoly();

    // set known points and values at those points and solve for coefficients
    void solve(const std::vector<T>& pts, const std::vector<T>& vals);
    //void solve(const Eigen::Matrix<T>& pts, const Eigen::Matrix<T, Eigen::Dynamic, 1>& vals);

    // do the same, but with weights

    // evaluate the function at the set of inputs
    T eval(const std::vector<T>& xtest);
    //T eval(const Eigen::Matrix<T, Eigen::Dynamic, 1>& xtest);
    //T eval(const Eigen::Matrix<T, NDIM, 1>& xtest);

private:

    // number of columns in the powers and coefficients arrays (determined from dims and order)
    static constexpr int32_t numBasis = getNumBasis(NDIM, NORDER);

    // powers of the the basis functions
    VectorBasis<numBasis*NDIM> powers;

    // A matrix for the problem
    Eigen::Matrix<T, Eigen::Dynamic, numBasis> A;

    // 1D array of the coefficients of each basis function
    Eigen::Matrix<T, numBasis, 1> coefficients;
};

template <class T, int32_t NDIM, int32_t NORDER>
WLSPoly<T,NDIM,NORDER>::WLSPoly() {
    // reserve space?
    // compute powers
    powers = getBasisPowers<NDIM,NORDER>();
}

//
// Accepts points and solution at those points, and computes coefficients for the polynomials
//
template <class T, int32_t NDIM, int32_t NORDER>
void WLSPoly<T,NDIM,NORDER>::solve(const std::vector<T>& pts,
                                   const std::vector<T>& soln) {

    // make sure inputs line up
    const size_t numPoints = soln.size();
    //assert(numPoints == NDIM*pts.size());

    // set up the A matrix from the given points
    std::vector<T> xs(NDIM);
    std::vector<T> vals(numBasis);
    for (size_t ii=0; ii<numPoints; ++ii) {
        // this is from basisEvals
        std::copy(pts.begin()+NDIM*ii, pts.begin()+NDIM*(ii+1), xs.begin());
        std::fill(vals.begin(), vals.end(), (T)1.0);
        for (size_t bb=0; bb<numBasis; ++bb) {
            for (size_t dd=0; dd<NDIM; ++dd) {
                if (powers[numBasis*dd + bb] > 0) {  //-1 means the power doesn't have that term
                    // weight goes in here, I think?
                    vals[bb] *= std::pow(xs[dd], powers[numBasis*dd + bb]);
                } else { // if(powers[numBasis*dd + bb] == 0) {  a 0th power is always 1, so just leave it as 1
                    ; // outVals[bb] = xs[dd];
                }
            }
        }
        for (size_t jj=0; jj<numBasis; ++jj) {
            A(ii,jj) = vals[jj];
        }
    }

    // solve it
    //coefficients = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(soln);
}


template <class T, int32_t NDIM, int32_t NORDER>
T WLSPoly<T,NDIM,NORDER>::eval(const std::vector<T>& xtest) {
    return 0.0;
}

