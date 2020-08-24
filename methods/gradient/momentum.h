#ifndef PROJEKATC___MOMENTUM_H
#define PROJEKATC___MOMENTUM_H

#include "../base_method.h"

namespace opt {
namespace method {
namespace gradient {

template<class real>
class momentum : public base_method<real> {
public:
    momentum() : base_method<real>() {}
    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();

        this->tic();

        real f_curr = f(x);
        real f_prev = f_curr + 1;

        arma::Col<real> gr = f.gradient(x);

        while (arma::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(f_curr);
            ls.set_current_f_val(f_curr);
            ls.set_current_g_val(gr);

            arma::Col<real> p = -gr;
            p = p * 0.9 - gr * 0.1;
            x += p * ls(f, x, p);

            f_prev = f_curr;
            f_curr = ls.get_current_f_val();
            gr = ls.get_current_g_val();
        }

        this->toc();
        this->f_min = f(x);
        this->gr_norm = arma::norm(gr);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
};

}
}
}

#endif //PROJEKATC___MOMENTUM_H
