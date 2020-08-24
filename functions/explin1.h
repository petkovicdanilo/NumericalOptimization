#ifndef PROJEKATC___EXPLIN1_H
#define PROJEKATC___EXPLIN1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class explin1 {
public:
    static real func(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "explin1: n must be positive";
        real z = 0;
        for (size_t i=0; i<v.size()-1; i++)
            z += exp(0.1 * v[i] * v[i+1]);
        for (size_t i=0; i<v.size(); i++)
            z -= v[i] * 10 * (i+1);
        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "explin1: n must be positive";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(v.size());
        for (size_t i=0; i<v.size()-1; i++) {
            z[i] += exp(v[i]*v[i+1] / 10) * v[i+1] / 10;
            z[i+1] += exp(v[i]*v[i+1] / 10) * v[i] / 10;
        }
        for (size_t i=0; i<v.size(); i++)
            z[i] -= (real)10*(i+1);
        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "explin1: n must be positive";
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(v.size(), v.size());
        for (size_t i=0; i<v.size(); i++) {
            real b = v[i];
            if (i > 0) {
                real a = v[i-1];
                z(i, i) += a*a*exp(0.1*a*b);
            }
            if (i+1 < v.size()) {
                real c = v[i+1];
                z(i, i) += c*c*exp(0.1*b*c);
            }
        }

        for (size_t i=0; i<v.size()-1; i++) {
            real a = v[i];
            real b = v[i+1];
            z(i, i+1) = (10+a*b)*exp(0.1*a*b);
            z(i+1, i) = z(i, i+1);
        }

        return z / (real)100;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n == 0)
            throw "explin1: n must be positive";

        return arma::zeros<arma::Col<real>>(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //PROJEKATC___EXPLIN1_H
