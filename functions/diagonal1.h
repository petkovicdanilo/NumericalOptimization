#ifndef NUMERICALOPTIMIZATION_DIAGONAL1_H
#define NUMERICALOPTIMIZATION_DIAGONAL1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class diagonal1 {
public:
    static real func(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        real z = 0;
        for (size_t i=0; i<v.size(); ++i) {
            real t = exp(v[i]) - (i+1)*v[i];
            z += t;
        }
        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(v.size());
        for (size_t i=0; i<v.size(); ++i) {
            z[i] = exp(v[i]) - (i+1);
        }
        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "diagonal1: n must be positive";
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(v.size(), v.size());
        for (size_t i=0; i<v.size(); ++i) {
            z(i, i) = exp(v[i]);
        }

        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n == 0)
            throw "diagonal1: n must be positive";

        return (1.0 / n) * arma::ones<arma::Col<real>>(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_DIAGONAL1_H
