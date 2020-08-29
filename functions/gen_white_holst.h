#ifndef NUMERICALOPTIMIZATION_GEN_WHITE_HOLST
#define NUMERICALOPTIMIZATION_GEN_WHITE_HOLST

#include "function.h"

namespace opt {
namespace function {

template<class real>
class gen_white_holst {
    static const int c = 100;
public:
    static real func(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "gen_white_holst: n must be positive";
        }

        real val = 0;

        size_t i = 0;
        while(i <= n-2) {
            val += c*(v[i+1]-v[i]*v[i]*v[i])*(v[i+1]-v[i]*v[i]*v[i]) + (1 - v[i])*(1 - v[i]);
            i++;
        }

        return val;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "gen_white_holst: n must be positive";
        }

        arma::Col<real> grad = arma::zeros<arma::Col<real>>(n);

        grad[0] = -2*(1 - v[0]) - 6*c*v[0]*v[0]*(-v[0]*v[0]*v[0] + v[1]);
        for(size_t i = 1; i <= n-2; i++) {
            grad[i] = -2*(1 - v[i]) - 6*c*v[i]*v[i]*(-v[i]*v[i]*v[i] + v[i+1])
                + 2*c*(-v[i-1]*v[i-1]*v[i-1] + v[i]);
        }
        grad[n-1] = 2*c*(-v[n-2]*v[n-2]*v[n-2] + v[n-1]);

        return grad;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        size_t n = v.size();
        arma::Mat<real> hes = arma::zeros<arma::Mat<real>>(n, n);

        if(n == 0) {
            throw "gen_white_holst: n must be positive";
        }

        for(size_t i = 0; i <= n-2; i++) {
            hes(i, i+1) = -6*c*v[i]*v[i];
            hes(i+1, i) = -6*c*v[i]*v[i];
            hes(i, i) = 2 + 2*c + 18*c*v[i]*v[i]*v[i]*v[i] - 12*c*v[i]*(-v[i]*v[i]*v[i] + v[i+1]);
        }

        hes(0, 0) -= 2*c;
        hes(n-1, n-1) = 2*c;

        return hes;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if(n == 0) {
            throw "gen_white_holst: n must be positive";
        }

        arma::Col<real> arr = { -1.2, 1 };
        arma::Col<real> col = arma::repmat(arr, n / 2 + 1, 1);
        col.resize(n);
        return col;
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};
}
}



#endif //NUMERICALOPTIMIZATION_GEN_WHITE_HOLST
