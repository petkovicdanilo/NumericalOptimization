#ifndef GOLDSTEIN_PRICE_H_INCLUDED
#define GOLDSTEIN_PRICE_H_INCLUDED

#include "../base_method.h"

namespace opt::method::modified_newton {

template<class real>
class goldstein_price : public base_method<real> {
public:
    goldstein_price(real eta = 0.2) : base_method<real>(), eta(eta) {}
    goldstein_price(real eta, real epsilon) : base_method<real>(epsilon), eta(eta) {}
    goldstein_price(real eta, real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter), eta(eta) {}
    goldstein_price(real eta, real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision), eta(eta) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();

        this->tic();

        real f_curr = f(x);
        real f_prev = f_curr + 1;

        auto gr = f.gradient(x);
        auto hes = f.hessian(x);

        while (arma::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(f_curr);
            ls.set_current_f_val(f_curr);
            ls.set_current_g_val(gr);

            arma::Col<real> d;
            bool solved = arma::solve(d, -hes, gr);

            if(arma::dot(d, -gr) / (arma::norm(d) * arma::norm(gr)) < eta || !solved) {
                d = -gr;
            }

            x += d * ls(f, x, d);

            f_prev = f_curr;
            f_curr = ls.get_current_f_val();
            gr = ls.get_current_g_val();
            hes = f.hessian(x);
        }

        this->toc();
        this->f_min = f_curr;
        this->gr_norm = arma::norm(gr);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }

protected:
    real eta;
};

}

#endif
