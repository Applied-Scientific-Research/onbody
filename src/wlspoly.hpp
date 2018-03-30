//
// A templated class for weighted least squares polynomials
//
// templatized on data type, number of spatial dimensions, and polynomial order
//

#include <vector>
#include <Eigen/Dense>

template <class T, int NDIM, int NORDER>
class WLSPoly {
public:

    // ctor computes some properties and readies the arrays
    WLSPoly();

    // set known points and values at those points and solve for coefficients
    void solve(const std::vector<T>& pts, const std::vector<T>& vals);
    void solve(const Eigen::Matrix<T>& pts, const Eigen::Matrix<T, Eigen::Dynamic, 1>& vals);

    // do the same, but with weights

    //size_t n;
    // state
    //alignas(32) std::vector<S> x;
    //alignas(32) std::vector<S> y;

    // evaluate the function at the set of inputs
    T eval(const std::vector<T>& xtest);
    T eval(const Eigen::Matrix<T, Eigen::Dynamic, 1>& xtest);
    T eval(const Eigen::Matrix<T, NDIM, 1>& xtest);

private:

    // number of columns in the powers and coefficients arrays
    // determined from dims and order
    int numBasis;

    // powers of the the basis functions
    Eigen::Matrix<T, Eigen::Dynamic, 1> powers;

    // A matrix for the problem
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A;

    // 1D array of the coefficients of each basis function
    Eigen::Matrix<T, Eigen::Dynamic, 1> coefficients;
};

template <class T, int NDIM, int NORDER>
WLSPoly<T,NDIM,NORDER>::WLSPoly() {
    numBasis = 6;
    // reserve space
    // compute powers
}

template <class T, int NDIM, int NORDER>
void WLSPoly<T,NDIM,NORDER>::solve(const std::vector<T>& pts,
                                   const std::vector<T>& vals) {

    // set up the A matrix from the given points

    // solve it
    coefficients = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(vals);
}


template <class T, int NDIM, int NORDER>
T WLSPoly<T,NDIM,NORDER>::eval(const std::vector<T>& xtest) {
    return 0.0;
}

