#ifndef NUMERICALOPTIMIZATION_BDQRTIC
#define NUMERICALOPTIMIZATION_BDQRTIC

#include "function.h"

namespace opt {
namespace function {

template<class real>
class bdqrtic {
public:
    static real func(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "bdqrtic: n must be positive";
        }

        if(n < 7) {
            throw "bdqrtic: n must be at least 7";
        }

        real val = 0;

        for (size_t i = 0; i <= n-5; i++) {
            val += (-4*v[i] + 3)*(-4*v[i] + 3) +
                (v[i]*v[i] + 2*v[i+1]*v[i+1] + 3*v[i+2]*v[i+2] + 4*v[i+3]*v[i+3] + 5*v[n-1]*v[n-1])*
                (v[i]*v[i] + 2*v[i+1]*v[i+1] + 3*v[i+2]*v[i+2] + 4*v[i+3]*v[i+3] + 5*v[n-1]*v[n-1]);
        }

        return val;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "bdqrtic: n must be positive";
        }

        if(n < 7) {
            throw "bdqrtic: n must be at least 7";
        }

        arma::Col<real> grad = arma::zeros<arma::Col<real>>(n);

        real ba0 = big_addend(0, v, n);
        real ba1 = big_addend(1, v, n);
        real ba2 = big_addend(2, v, n);
        real ba3 = big_addend(3, v, n);

        grad[0] = -8*(-4*v[0] + 3) + 4*v[0]*ba0;
        grad[1] = 8*v[1]*ba0 - 8*(-4*v[1] + 3) + 4*v[1]*ba1;
        grad[2] = 12*v[2]*ba0 + 8*v[2]*ba1 + 4*v[2]*ba2 - 8*(-4*v[2] + 3);
        grad[n-1] = 20*v[n-1]*ba0 + 20*v[n-1]*ba1 + 20*v[n-1]*ba2;

        for(size_t i = 3; i <= n-5; i++) {
            grad[i] = 16*v[i]*ba0 + 12*v[i]*ba1 + 8*v[i]*ba2 + 4*v[i]*ba3 - 8*(-4*v[i] + 3);
            grad[n-1] += 20*v[n - 1]*ba3;

            ba0 = ba1;
            ba1 = ba2;
            ba2 = ba3;
            ba3 = big_addend(i+1, v, n);
        }

        grad[n-4] = 16*v[n-4]*ba0 + 12*v[n-4]*ba1 + 8*v[n-4]*ba2;
        grad[n-3] = 16*v[n-3]*ba1 + 12*v[n-3]*ba2;
        grad[n-2] = 16*v[n-2]*ba2;

        return grad;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        size_t n = v.size();

        if(n == 0) {
            throw "bdqrtic: n must be positive";
        }

        if(n < 7) {
            throw "bdqrtic: n must be at least 7";
        }

        arma::Mat<real> hes = arma::zeros<arma::Mat<real>>(n, n);

        real ba0 = big_addend(0, v, n);
        real ba1 = big_addend(1, v, n);
        real ba2 = big_addend(2, v, n);
        real ba3 = big_addend(3, v, n);

        hes(0, 0) = 32 + 8*v[0]*v[0] + 4*ba0;
        hes(0, 1) = 16*v[0]*v[1];
        hes(0, 2) = 24*v[0]*v[2];
        hes(0, 3) = 32*v[0]*v[3];
        hes(0, n-1) += 40*v[0]*v[n-1];

        hes(1, 0) = 16*v[0]*v[1];
        hes(1, 1) = 32 + 40*v[1]*v[1] + 8*ba0 + 4*ba1;
        hes(1, 2) = 64*v[1]*v[2];
        hes(1, 3) = 88*v[1]*v[3];
        hes(1, 4) = 32*v[1]*v[4];
        hes(1, n-1) += 120*v[1]*v[n-1];

        hes(2, 0) = 24*v[2]*v[0];
        hes(2, 1) = 64*v[1]*v[2];
        hes(2, 2) = 112*v[2]*v[2] + 12*ba0 + 8*ba1 + 4*ba2 + 32;
        hes(2, 3) = 160*v[2]*v[3];
        hes(2, 4) = 88*v[2]*v[4];
        hes(2, 5) = 32*v[2]*v[5];
        hes(2, n-1) += 240*v[2]*v[n-1];

        hes(n-1, 0) = 40*v[0]*v[n-1];
        hes(n-1, 1) = 120*v[1]*v[n-1];
        hes(n-1, 2) = 240*v[2]*v[n-1];
        hes(n-1, 3) = 360*v[3]*v[n-1];
        hes(n-1, 4) = 280*v[4]*v[n-1];
        hes(n-1, 5) = 160*v[5]*v[n-1];
        hes(n-1, n-1) = 20*(ba0 + ba1 + ba2) + 600*v[n-1]*v[n-1];

        for(size_t i = 3; i <= n-5; i++) {
            hes(i, i-3) = 32*v[i]*v[i-3];
            hes(i, i-2) = 88*v[i]*v[i-2];
            hes(i, i-1) = 160*v[i]*v[i-1];
            hes(i, i) = 240*v[i]*v[i] + 16*ba0 + 12*ba1 + 8*ba2 + 4*ba3 + 32;
            hes(i, i+1) = 160*v[i]*v[i+1];
            hes(i, i+2) = 88*v[i]*v[i+2];
            hes(i, i+3) = 32*v[i]*v[i+3];
            hes(i, n-1) += 400*v[i]*v[n-1];

            hes(n-1, i) += 40*v[i]*v[n-1];
            hes(n-1, i+1) += 80*v[i+1]*v[n-1];
            hes(n-1, i+2) += 120*v[i+2]*v[n-1];
            hes(n-1, i+3) += 160*v[i+3]*v[n-1];
            hes(n-1, n-1) += 20*ba3 + 200*v[n-1]*v[n-1];

            ba0 = ba1;
            ba1 = ba2;
            ba2 = ba3;
            ba3 = big_addend(i+1, v, n);
        }

        hes(n-4, n-7) += 32*v[n-4]*v[n-7];
        hes(n-4, n-6) += 88*v[n-4]*v[n-6];
        hes(n-4, n-5) += 160*v[n-4]*v[n-5];
        // hes(n-4, n-4) += 232*v[n-4]*v[n-4] + 16*ba1 + 12*ba2 + 8*ba3;
        hes(n-4, n-4) += 232*v[n-4]*v[n-4] + 16*ba0 + 12*ba1 + 8*ba2;
        hes(n-4, n-3) += 144*v[n-4]*v[n-3];
        hes(n-4, n-2) += 64*v[n-4]*v[n-2];
        hes(n-4, n-1) += 360*v[n-4]*v[n-1];

        hes(n-3, n-6) += 32*v[n-3]*v[n-6];
        hes(n-3, n-5) += 88*v[n-3]*v[n-5];
        hes(n-3, n-4) += 144*v[n-3]*v[n-4];
        // hes(n-3, n-3) += 200*v[n-3]*v[n-3] + 16*ba2 + 12*ba3;
        hes(n-3, n-3) += 200*v[n-3]*v[n-3] + 16*ba1 + 12*ba2;
        hes(n-3, n-2) += 96*v[n-3]*v[n-2];
        hes(n-3, n-1) += 280*v[n-3]*v[n-1];

        hes(n-2, n-5) += 32*v[n-2]*v[n-5];
        hes(n-2, n-4) += 64*v[n-2]*v[n-4];
        hes(n-2, n-3) += 96*v[n-2]*v[n-3];
        // hes(n-2, n-2) += 128*v[n-2]*v[n-2] + 16*ba3;
        hes(n-2, n-2) += 128*v[n-2]*v[n-2] + 16*ba2;
        hes(n-2, n-1) += 160*v[n-2]*v[n-1];

        return hes;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if(n == 0) {
            throw "bdqrtic: n must be positive";
        }

        return arma::ones<arma::Col<real>>(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
private:
    static real big_addend(size_t i, const arma::Col<real>& v, size_t n) {
        return v[i]*v[i] + 2*v[i+1]*v[i+1] + 3*v[i+2]*v[i+2] + 4*v[i+3]*v[i+3] + 5*v[n-1]*v[n-1];
    }
};
}
}

#endif //NUMERICALOPTIMIZATION_BDQRTIC
