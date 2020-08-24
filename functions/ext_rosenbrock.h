#ifndef EXTENDED_ROSENBROCK_H_INCLUDED
#define EXTENDED_ROSENBROCK_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class ext_rosenbrock {
public:
    static const int c = 100;

    static real func(const arma::Col<real>& v) {
        if (v.size() % 2 || v.size() == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        size_t n = v.size();
        real z = 0.0;

        for (size_t i = 0; i < n; i += 2) {
            real t = v[i+1] - v[i]*v[i];
            z += c * t*t;
            t = 1 - v[i];
            z += t*t;
        }

        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() % 2 || v.size() == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        size_t n = v.size();
        arma::Col<real> z = arma::zeros<arma::Mat<real>>(n);

        for (size_t i = 0; i < n; i++) {
            // i & 1 != 0 <=> i is odd
            z[i] = i & 1 ? c * (2*v[i] - 2*v[i-1]*v[i-1])
                         : c * (4*v[i]*v[i]*v[i] - 4*v[i+1]*v[i]) + 2*v[i] - 2;
        }

        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() % 2 || v.size() == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        size_t n = v.size();
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(n, n);

        for (size_t i = 0; i < n; i++) {
            // i & 1 != 0 <=> i is odd
            if (i & 1) {
                z(i, i-1) = -4*c*v[i-1];
                z(i, i) = 2*c;
            } else {
                z(i, i) = 12*c*v[i]*v[i] - 4*c*v[i+1] + 2;
                z(i, i+1) = -4*c*v[i];
            }
        }

        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n % 2 || n == 0) {
            throw "ext_rosenbrock: n must be even and positive";
        }

        arma::Col<real> z = arma::zeros<arma::Col<real>>(n);

        for (size_t i = 0; i < n; i += 2) {
            z[i] = -1.2;
            z[i+1] = 1;
        }

        return z;
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

template<class real>
const int ext_rosenbrock<real>::c;

}
}

#endif // EXTENDED_ROSENBROCK_H_INCLUDED
