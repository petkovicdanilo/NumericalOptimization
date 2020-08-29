#ifndef NUMERICALOPTIMIZATION_EXT_HIEBERT
#define NUMERICALOPTIMIZATION_EXT_HIEBERT

#include "function.h"

namespace opt {
namespace function {

template<class real>
class ext_hiebert {
    static const int c1 = 10;
    static const int c2 = 50000;
public:
    static real func(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "ext_hiebert: n must be positive";
        }

        real val = 0;

        for(size_t i = 0; i < (n-1)/2; i++) {
            val += (v[2*i] - c1)*(v[2*i] - c1)
                + (v[2*i]*v[2*i + 1] - c2)*(v[2*i]*v[2*i + 1] - c2);
        }

        return val;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "ext_hiebert: n must be positive";
        }

        arma::Col<real> grad = arma::zeros<arma::Col<real>>(n);

        for(size_t i = 0; i < (n-1)/2; i++) {
            grad[2*i] = 2*(v[2*i] - c1) + 2*v[2*i + 1]*(v[2*i]*v[2*i + 1] - c2);
            grad[2*i + 1] = 2*v[2*i]*(v[2*i]*v[2*i + 1] - c2);
        }

        return grad;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "ext_hiebert: n must be positive";
        }

        arma::Mat<real> hes = arma::zeros<arma::Mat<real>>(n, n);

        for(size_t i = 0; i < (n-1)/2; i++) {
            hes(2*i, 2*i) = 2 + 2*v[2*i + 1]*v[2*i + 1];
            hes(2*i, 2*i + 1) = 4*v[2*i]*v[2*i + 1] - 2*c2;

            hes(2*i + 1, 2*i) = 4*v[2*i]*v[2*i + 1] - 2*c2;
            hes(2*i + 1, 2*i + 1) = 2*v[2*i]*v[2*i];
        }

        return hes;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if(n == 0) {
            throw "ext_hiebert: n must be positive";
        }

        return arma::zeros<arma::Col<real>>(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};
}
}

#endif //NUMERICALOPTIMIZATION_EXT_HIEBERT
