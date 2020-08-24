#ifndef NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H
#define NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class gen_psc1 {
public:
    static real func(const arma::Col<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        real z = 0;
        for (size_t i=0; i<v.size()-1; ++i) {
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
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(v.size());
        auto n = v.size();
        real prev = 0;
        z[0]= 2* (2*v[0]+v[1]) * (v[0]*v[0] +v[0]*v[1]+v[1]*v[1]) + 2*cos(v[0])*sin(v[0]);
        z[n-1] = 2*(v[n-2]+2*v[n-1])*(v[n-2]*v[n-2]+v[n-2]*v[n-1]+v[n-1]*v[n-1])-2*cos(v[n-1])*sin(v[n-1]);;
        for (size_t i=1; i<n-1; ++i) {
            real t1 = v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1];
            real t2 = v[i-1]*v[i-1] + v[i]*v[i] + v[i-1]*v[i];
            z[i] += 2*t1*(2*v[i] + v[i+1]) + 2*t2*(2*v[i] + v[i-1]);
        }
        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() < 2)
            throw "gen_psc1: n must be greater than 1";
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(v.size(), v.size());
        size_t n = v.size();
        for (size_t i=0; i<v.size()-1; ++i) {
            // d^2 vi

            if (i == 0){
                z(i, i) = 2*(2*v[0]+v[1])*(2*v[0]+v[1]) + 4*(v[0]*v[0]+v[0]*v[1]+v[1]*v[1]) + 2*cos(v[0])*cos(v[0]) - 2*sin(v[0])*sin(v[0]);
            }else{
                z(i, i) = 2*(v[i-1]+2*v[i])*(v[i-1]+2*v[i]) + 4*(v[i-1]*v[i-1]+v[i-1]*v[i]+v[i]*v[i]) + 2*(2*v[i]+v[i+1])*(2*v[i]+v[i+1]) + 4*(v[i]*v[i]+v[i]*v[i+1]+v[i+1]*v[i+1]);
            }

            if(i+1<v.size()){
                // d vivi+1
                z(i+1, i) = 2*( (2*v[i] + v[i+1])*(2*v[i+1] + v[i]) + (v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1]) );
                // d vi+1vi
                z(i, i+1) = 2*( (2*v[i] + v[i+1])*(2*v[i+1] + v[i]) + (v[i]*v[i] + v[i+1]*v[i+1] + v[i]*v[i+1]) );
            }
        }

        z(n-1, n-1) = 2*(v[n-2]+2*v[n-1])*(v[n-2]+2*v[n-1]) + 4*(v[n-2]*v[n-2]+v[n-2]*v[n-1]+v[n-1]*v[n-1]) - 2*cos(v[n-1])*cos(v[n-1]) + 2*sin(v[n-1])*sin(v[n-1]);
        return z;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n < 2)
            throw "gen_psc1: n must be greater than 1";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(n);
        for (size_t i=0; i<n; ++i) {
            if (i % 2) {
                z[i] = 0.1;
            } else {
                z[i] = 3;
            }
        }
        return z;
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //NUMERICALOPTIMIZATION_GENERALIZED_PSC1_H
