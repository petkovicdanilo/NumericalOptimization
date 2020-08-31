#ifndef NEWTON_LINE_SEARCH_H_INCLUDED
#define NEWTON_LINE_SEARCH_H_INCLUDED

#include "../base_method.h"

namespace opt::method::newton {

template<class real>
class newton_line_search : public base_method<real> {
public:
    newton_line_search() : base_method<real>(){}
    newton_line_search(real epsilon) : base_method<real>(epsilon){}
    newton_line_search(real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter) {}
    newton_line_search(real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision) {}

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

            arma::Col<real> d = arma::solve(-hes, gr);
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
};

}

#endif
