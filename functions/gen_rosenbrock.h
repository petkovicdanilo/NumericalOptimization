#ifndef GENERALIZED_ROSENBROCK_H_INCLUDED
#define GENERALIZED_ROSENBROCK_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class gen_rosenbrock {
public:
    static const int c = 100;

    static real func(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        size_t m = v.size() - 1;
        real z = 0.0;

        for (size_t i = 0; i < m; ++i) {
            real t = v[i+1] - v[i]*v[i];
            z += c * t*t;
            t = 1 - v[i];
            z += t*t;
        }

        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        arma::Col<real> z = arma::zeros<arma::Mat<real>>(n);

        z[0] = c * 4 * (v[0]*v[0]*v[0] - v[0]*v[1]) - 2*(1-v[0]);
        for (size_t i = 1; i < m; ++i) {
            z[i] = c * (4*v[i]*v[i]*v[i] - 2*v[i-1]*v[i-1] - 4*v[i+1]*v[i] + 2*v[i]) - 2*(1-v[i]);
        }
        z[m] = c * 2 * (v[m] - v[m-1]*v[m-1]);

        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(n, n);

        // computes first row
        z(0, 0) = 2 + 8*c*v[0]*v[0] - 4*c*(-v[0]*v[0] + v[1]);
        z(0, 1) = -4*c*v[0];

        // computes all rows except for first and last
        for (size_t i = 1; i < m; i++) {
            z(i, i-1) = -4*c*v[i-1];
            z(i, i) = 2 + 2*c + 8*c*v[i]*v[i] - 4*c*(-v[i]*v[i] + v[i+1]);
            z(i, i+1) = -4*c*v[i];
        }

        // computes last row
        z(m, m) = 2*c;
        z(m, m-1) = -4*c*v[m-1];

        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "gen_rosenbrock: n must be positive";
        }

        arma::Col<real> z = arma::zeros<arma::Col<real>>(n);

        for (size_t i = 0; i < n; ++i) {
            z[i] = i & 1 ? 1 : -1.2; // i & 1 != 0 <=> i is odd
        }

        return z;
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

template<class real>
const int gen_rosenbrock<real>::c;

}
}

#endif // GENERALIZED_ROSENBROCK_H_INCLUDED
