#ifndef NUMERICALOPTIMIZATION_SR1_H
#define NUMERICALOPTIMIZATION_SR1_H

#include "../base_method.h"

namespace opt {
namespace method {
namespace quasi_newton {

template<class real>
class sr1 : public base_method<real> {
public:
    sr1() : base_method<real>(), r(1e-8) {}
    sr1(real r) : base_method<real>(), r(r) {}
    sr1(real r, real epsilon) : base_method<real>(epsilon), r(r) {}
    sr1(real r, real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter), r(r) {}
    sr1(real r, real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision), r(r) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();

        this->tic();

        size_t n = x.size();

        arma::Col<real> x0;
        arma::Col<real>& x1 = x;
        arma::Col<real> gradient_prev;
        arma::Col<real> gragient_curr = f.gradient(x1);

        arma::Mat<real> H = arma::eye(n, n);

        real fcur = f(x1);
        real fprev = fcur + 1;

        while (arma::norm(gragient_curr) > this->epsilon && this->iter_count < this->max_iter && fabs(fprev-fcur)/(1+fabs(fcur)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(fcur);
            ls.set_current_f_val(fcur);
            ls.set_current_g_val(gragient_curr);

            arma::Col<real> direction = -1 * (H * gragient_curr);

            fprev = fcur;
            x0 = x1;
            gradient_prev = gragient_curr;

            real t = ls(f,x1,direction);
            x1 += direction * t;

            fcur = ls.get_current_f_val();
            gragient_curr = ls.get_current_g_val();

            auto s = x1 - x0;
            auto y = gragient_curr - gradient_prev;

            auto tmp_s_H_dot_y = s - H * y;

            if (arma::dot(tmp_s_H_dot_y, y) >= r * arma::norm(y) * arma::norm(tmp_s_H_dot_y) ) {
                H += (tmp_s_H_dot_y * tmp_s_H_dot_y.t()) / arma::dot(tmp_s_H_dot_y, y);
            }

        }

        this->toc();
        this->f_min = fcur;
        this->gr_norm = arma::norm(gragient_curr);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
protected:
    real r; // coef which determines whether or not to update H
};

}
}
}

#endif //NUMERICALOPTIMIZATION_SR1_H
