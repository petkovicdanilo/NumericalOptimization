#ifndef CUBE_H_INCLUDED
#define CUBE_H_INCLUDED

#include "function.h"

namespace opt {
namespace function {

template<class real>
class cube {
public:
    static real func(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "cube: n must be positive";
        }

        size_t n = v.size();
        real z = (v[0] - 1)*(v[0] - 1);

        for (size_t i = 1; i < n; ++i) {
            real t = v[i] - v[i-1]*v[i-1]*v[i-1];
            z += 100*t*t;
        }

        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "cube: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        arma::Col<real> z = arma::zeros<arma::Col<real>>(n);

        z[0] = -2*(1 - v[0]) - 600*v[0]*v[0]*(-v[0]*v[0]*v[0] + v[1]);
        for (size_t i = 1; i < m; ++i) {
            z[i] = -600*v[i]*v[i]*(-v[i]*v[i]*v[i] + v[i+1]) + 200*(-v[i-1]*v[i-1]*v[i-1] + v[i]);
        }
        z[m] = 200*(-v[m-1]*v[m-1]*v[m-1] + v[m]);

        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "cube: n must be positive";
        }

        size_t n = v.size();
        size_t m = n - 1;
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(n, n);

        for (size_t i = 0; i < m; ++i) {
            z(i, i) = 200 + 1800*v[i]*v[i]*v[i]*v[i] - 1200*v[i]*(-v[i]*v[i]*v[i] + v[i+1]);
            z(i, i+1) = z(i+1, i) = -600*v[i]*v[i];
        }
        z(0, 0) -= 198;
        z(m, m) = 200;

        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "cube: n must be positive";
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

}
}

#endif // CUBE_H_INCLUDED
