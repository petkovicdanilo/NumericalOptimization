#ifndef NUMERICALOPTIMIZATION_AP_QUAD_H
#define NUMERICALOPTIMIZATION_AP_QUAD_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class almost_pert_quad {
public:
    static const int c = 100;

    static real func(const arma::Col<real>& v) {
        size_t n = v.size();
        if (n == 0)
            throw "almost_pert_quad: n must be positive";
        real z = 0;
        for (size_t i=0; i<n; ++i) {
            real t = (i+1)*v[i]*v[i];
            z += t;
        }
        z+=n*(v[0]+v[n-1])*(v[0]+v[n-1])/c;
        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        size_t n = v.size();
        if (n == 0)
            throw "almost_pert_quad: n must be positive";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(n);

        real t = (2.0/c)*(v[0]+v[n-1]);

        for (size_t i=0; i<n; ++i) {
            z[i] = 2*(i+1)*v[i];
        }
        z[0]+= n*t;
        z[n-1]+= n*t;
        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        size_t n = v.size();
        if (n == 0)
            throw "almost_pert_quad: n must be positive";
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(n, n);
        for (size_t i=0; i<n; ++i) {
            z(i, i) = 2*(i+1);
        }
        z(n-1, n-1) += (2.0*n)/c;
        z(0, 0) += (2.0*n)/c;
        z(n-1, 0) += (2.0*n)/c;
        z(0, n-1) += (2.0*n)/c;

        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n == 0)
            throw "almost_pert_quad: n must be positive";

        return 0.5 * arma::ones<arma::Col<real>>(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

template<class real>
const int almost_pert_quad<real>::c;
}
}

#endif //NUMERICALOPTIMIZATION_AP_QUAD_H
