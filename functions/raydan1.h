#ifndef RAYDAN1_H_INCLUDED
#define RAYDAN1_H_INCLUDED

#include <cmath>
#include "function.h"

namespace opt::function{

template<class real>
class raydan1 {
public:
    static real func(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        size_t n = v.size();
        real z = 0.0;

        for (size_t i = 0; i < n; ++i) {
            z += ((i+1) / 10.0) * (exp(v[i]) - v[i]);
        }

        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        size_t n = v.size();
        arma::Col<real> z = arma::zeros<arma::Col<real>>(n);

        for (size_t i = 0; i < n; ++i) {
            z[i] = ((i+1) / 10.0) * (exp(v[i]) - 1);
        }

        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() == 0) {
            throw "raydan1: n must be positive";
        }

        size_t n = v.size();
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(n, n);

        for (size_t i = 0; i < n; ++i) {
            z(i, i) = ((i+1) / 10.0) * exp(v[i]);
        }

        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n == 0) {
            throw "raydan1: n must be positive";
        }

        return arma::ones<arma::Col<real>>(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}

#endif // RAYDAN1_H_INCLUDED
