#ifndef LEVENBERG_MARQUARDT_H_INCLUDED
#define LEVENBERG_MARQUARDT_H_INCLUDED

#include "../base_method.h"

namespace opt::method::modified_newton {

template<class real>
class levenberg_marquardt : public base_method<real> {
public:
    levenberg_marquardt() : base_method<real>(){}
    levenberg_marquardt(real epsilon) : base_method<real>(epsilon){}
    levenberg_marquardt(real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter) {}
    levenberg_marquardt(real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();

        real lambda = 1e-4;
        real lambdaMax = 1e7;
        real lambdaMin = 1e-10;
        int lambdaMul = 10;

        size_t n = x.size();

        this->tic();

        real f_curr = f(x);
        real f_prev = f_curr + 1;

        auto gr = f.gradient(x);
        auto hes = f.hessian(x);

        while (arma::norm(gr) > this->epsilon && this->iter_count < this->max_iter &&
            fabs(f_prev-f_curr)/(1+fabs(f_curr)) > this->working_precision) {

            ++this->iter_count;

            arma::Mat<real> leftTerm = hes + (lambda * arma::diagmat(hes));
            arma::Col<real> d = -arma::solve(leftTerm, gr);

            auto xminCurr = x + d;
            auto valCurr = f(xminCurr);

            if(valCurr >= f_curr) {
                if(lambda == lambdaMax) {
                    break;
                }
                lambda = std::min(lambdaMul * lambda, lambdaMax);
            }
            else {
                x = xminCurr;
                f_prev = f_curr;
                f_curr = valCurr;
                gr = f.gradient(x);
                hes = f.hessian(x);

                lambda = std::max(lambda / lambdaMul, lambdaMin);
            }
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
