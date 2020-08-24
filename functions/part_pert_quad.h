#ifndef PROJEKATC___PP_QUAD_H
#define PROJEKATC___PP_QUAD_H

#include "function.h"

namespace opt {
namespace function {

template<class real>
class part_pert_quad {
public:
    static real func(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "part_pert_quad: n must be positive";
        real z = 0;
        z += v[0]*v[0];
        real ps = 0;
        for (size_t i=0; i<v.size(); i++) {
            ps += v[i];
            z += v[i]*v[i]*(i+1);
            z += ps*ps / 100;
        }
        return z;
    }

    static arma::Col<real> gradient(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "part_pert_quad: n must be positive";
        arma::Col<real> z = arma::zeros<arma::Col<real>>(v.size());
        arma::Col<real> ps(v.size());
        ps[0] = v[0];
        real t = 0;
        for (size_t i=1; i<v.size(); i++)
            ps[i] = ps[i-1] + v[i];
        for (size_t i=0; i<v.size(); i++)
            t += v[i] * (v.size() - i) * 2;

        z[0] = t / 100 + v[0] * 4;
        for (size_t i=1; i<v.size(); i++) {
            t -= 2*ps[i-1];
            z[i] = t / 100 + v[i] * (i+1) * 2;
        }
        return z;
    }

    static arma::Mat<real> hessian(const arma::Col<real>& v) {
        if (v.size() == 0)
            throw "part_pert_quad: n must be positive";
        arma::Mat<real> z = arma::zeros<arma::Mat<real>>(v.size(), v.size());
        for (size_t i=0; i<v.size(); i++) {
            for (size_t j=0; j<v.size(); j++) {
                if (i == j) {
                    if (i == 0)
                        z(i, j) += 200;
                    z(i, j) += 200*(i+1);
                }
                z(i, j) += 2*(v.size() - std::max(i, j));
            }
        }

        return z / (real)100;
    }

    static arma::Col<real> starting_point(const size_t n) {
        if (n == 0)
            throw "part_pert_quad: n must be positive";

        return 0.5 * arma::ones<arma::Col<real>>(n);
    }

    static function<real> get_function() {
        return function<real>(func, gradient, hessian, starting_point);
    }
};

}
}

#endif //PROJEKATC___PP_QUAD_H
