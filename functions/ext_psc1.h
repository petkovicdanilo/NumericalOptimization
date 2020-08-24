#ifndef PROJEKATC___EXTENDED_PSC1_H
#define PROJEKATC___EXTENDED_PSC1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class ext_psc1 {
public:
    static real func(const arma::Col<real>& v) {
        if (v.size() % 2 || v.size() == 0)
            throw "ext_psc1: n must be even and positive";
        real z = 0;
        for (size_t i=0; i<v.size(); i+=2) {
            real t = v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1];
            z += t*t;
            t = sin(v[i]);
            z += t*t;
            t = cos(v[i+1]);
            z += t*t;
        }
        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() % 2 || v.size() == 0)
            throw "ext_psc1: n must be even and positive";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(v.size());
        for (size_t i=0; i<v.size(); i+=2) {
            real t = v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1];
            z[i] += 2*t*(2*v[i] + v[i+1]);
            z[i] += 2*sin(v[i])*cos(v[i]);

            z[i+1] += 2*t*(2*v[i+1] + v[i]);
            z[i+1] -= 2*cos(v[i+1])*sin(v[i+1]);
        }
        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() % 2 || v.size() == 0)
            throw "ext_psc1: n must be even and positive";
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(v.size(), v.size());
        for (size_t i=0; i<v.size(); i+=2) {
            // 0-0
            real a = v[i];
            real b = v[i+1];

            z(i, i) = 2*(2*a+b)*(2*a+b) + 4*(a*a+a*b+b*b)
                      + 2*cos(a)*cos(a) - 2*sin(a)*sin(a);

            // 0-1
            z(i+1, i) = 2*(2*a+b)*(a+2*b) + 2*(a*a+a*b+b*b);
            z(i, i+1) = z(i+1, i);

            // 1-1
            z(i+1, i+1) = 2*(a+2*b)*(a+2*b) + 4*(a*a+a*b+b*b)
                          - 2*cos(b)*cos(b) + 2*sin(b)*sin(b);
        }

        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n % 2 || n == 0)
            throw "ext_psc1: n must be even and positive";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(n);
        for (size_t i=0; i<n; i+=2) {
            z[i] = 3;
            z[i+1] = 0.1;
        }
        return z;
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //PROJEKATC___EXTENDED_PSC1_H
