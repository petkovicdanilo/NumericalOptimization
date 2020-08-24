#ifndef NUMERICALOPTIMIZATION_BFGS_H
#define NUMERICALOPTIMIZATION_BFGS_H

#include "../base_method.h"

namespace opt {
namespace method {
namespace quasi_newton {

template<class real>
class bfgs : public base_method<real> {
public:
    bfgs() : base_method<real>() {}
    bfgs(real epsilon) : base_method<real>(epsilon) {}
    bfgs(real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter) {}
    bfgs(real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();

        this->tic();

        size_t n = x.size();

        arma::Col<real> x0;
        arma::Col<real>& x1 = x;
        arma::Col<real> gr0;
        arma::Col<real> gr1 = f.gradient(x1);

        arma::Mat<real> H = arma::eye(n, n);

        real fcur = f(x1);
        real fprev = fcur + 1;

        while (arma::norm(gr1) > this->epsilon && this->iter_count < this->max_iter && fabs(fprev-fcur)/(1+fabs(fcur)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(fcur);
            ls.set_current_f_val(fcur);
            ls.set_current_g_val(gr1);

            arma::Col<real> direction = (-1) * (H * gr1);

            fprev = fcur;
            x0 = x1;
            gr0 = gr1;

            real t = ls(f, x1, direction);
            x1 += direction * t;

            fcur = ls.get_current_f_val();
            gr1 = ls.get_current_g_val();

            auto s = x1 - x0;
            auto y = gr1 - gr0;

            real auxSc = arma::dot(s, y); // auxiliary variable
            auto H_dot_y = H * y;

            H +=  s*s.t()*(auxSc + arma::dot(y, H_dot_y))/(auxSc*auxSc) - ((H_dot_y * s.t()) + (s* y.t() * H))/auxSc;
        }

        this->toc();
        this->f_min = fcur;
        this->gr_norm = arma::norm(gr1);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
};

}
}
}

#endif //NUMERICALOPTIMIZATION_BFGS_H
