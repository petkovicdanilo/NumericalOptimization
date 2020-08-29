#ifndef NUMERICALOPTIMIZATION_EXT_TRIDIAG1
#define NUMERICALOPTIMIZATION_EXT_TRIDIAG1

#include "function.h"
#include<cmath>

namespace opt {
namespace function {

template<class real>
class ext_tridiag1 {
public:
    static real func(const arma::Col<real>& v) {
        size_t n = v.size();

        if (n == 0) {
            throw "ext_tridiag1: n must be positive";
        }

        real val = 0.0;

        for(size_t i = 0; i < n/2; i++) {
            val += (v[2*i] + v[2*i + 1] - 3)*(v[2*i] + v[2*i + 1] - 3)
                + pow(v[2*i] - v[2*i + 1] + 1, 4);
        }

        return val;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        size_t n = v.size();

        if (n == 0) {
            throw "ext_tridiag1: n must be positive";
        }

        arma::Col<real> grad = arma::zeros<arma::Col<real>>(n);

        for(size_t i = 0; i < n/2; i++) {
            grad[2*i] = 2*(v[2*i] + v[2*i + 1] - 3) + 4*pow((v[2*i] - v[2*i + 1] + 1), 3);
            grad[2*i + 1] = 2*(v[2*i] + v[2*i + 1] - 3) - 4*pow((v[2*i] - v[2*i + 1] + 1), 3);
        }

        return grad;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        size_t n = v.size();

        if (n == 0) {
            throw "ext_tridiag1: n must be positive";
        }

        arma::Mat<real> hes = arma::zeros<arma::Mat<real>>(n, n);

        for(size_t i = 0; i < n/2; i++) {
            real add_val = 12*(v[2*i] - v[2*i + 1] + 1)*(v[2*i] - v[2*i + 1] + 1);

            hes(2*i, 2*i) = 2 + add_val;
            hes(2*i, 2*i + 1) = 2 - add_val;
            hes(2*i + 1, 2*i) = 2 - add_val;
            hes(2*i + 1, 2*i + 1) = 2 + add_val;
        }

        return hes;
    }

    static arma::Col<real> starting_point(const size_t n) {
        return 2 * arma::ones(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
private:
    static arma::Col<real> get_inner_sum(int i, const arma::Col<real>& v) {
        real inner_sum = 0.0;
        size_t n = v.size();

        for(size_t j = 0; j < n; j++) {
            inner_sum += i*j*v[j] - 1;
        }

        return inner_sum;
    }
};
}
}

#endif //NUMERICALOPTIMIZATION_EXT_TRIDIAG1
