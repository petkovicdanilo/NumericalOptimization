//
// Created by lazar on 25.6.19..
//

#ifndef NUMERICALOPTIMIZATION_L_BFGS_H
#define NUMERICALOPTIMIZATION_L_BFGS_H


#include "../base_method.h"
#include <list>

namespace opt {
namespace method {
namespace quasi_newton {

template<class real>
class l_bfgs : public base_method<real> {
private:
    size_t cache_size;
public:
    l_bfgs() : base_method<real>(), cache_size(5) {}
    l_bfgs(size_t cache_size) : base_method<real>(), cache_size(cache_size) {}
    l_bfgs(size_t cache_size, real epsilon) : base_method<real>(epsilon), cache_size(cache_size) {}
    l_bfgs(size_t cache_size, real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter), cache_size(cache_size) {}
    l_bfgs(size_t cache_size, real epsilon, size_t max_iter, real working_precision) : base_method<real>(epsilon, max_iter, working_precision), cache_size(cache_size) {}

    void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x) {
        this->iter_count = 0;
        ls.clear_f_vals();

        this->tic();

        arma::Col<real> x0;
        arma::Col<real>& x1 = x;
        arma::Col<real> gradient_prev;
        arma::Col<real> gradient_curr = f.gradient(x1);

        real H = 1;

        real fcur = f(x1);
        real fprev = fcur + 1;

        std::list<arma::Col<real>> s_cache, y_cache;
        std::list<real> rho_cache;


        while (arma::norm(gradient_curr) > this->epsilon && this->iter_count < this->max_iter && fabs(fprev-fcur)/(1+fabs(fcur)) > this->working_precision) {
            ++this->iter_count;
            ls.push_f_val(fcur);
            ls.set_current_f_val(fcur);
            ls.set_current_g_val(gradient_curr);

            arma::Col<real> direction = this->two_loop_recursion(H,gradient_curr,s_cache,y_cache,rho_cache);

            fprev = fcur;
            x0 = x1;
            gradient_prev = gradient_curr;

            real t = ls(f,x1,direction);
            x1 += direction * t;

            fcur = ls.get_current_f_val();
            gradient_curr = ls.get_current_g_val();

            arma::Col<real> s = x1 - x0;
            this->add_to_cache(s,s_cache);
            arma::Col<real> y = gradient_curr - gradient_prev;
            this->add_to_cache(y,y_cache);

            real rho = 1/(arma::dot(s, y));
            this->add_to_cache(rho,rho_cache);

            H = (arma::dot(s, y))/(arma::dot(y, y));
        }

        this->toc();
        this->f_min = fcur;
        this->gr_norm = arma::norm(gradient_curr);
        this->f_call_count = f.get_call_count();
        this->g_call_count = f.get_grad_count();
        this->h_call_count = f.get_hess_count();
    }
private:
    arma::Col<real> two_loop_recursion(real H, arma::Col<real>& gradient, std::list<arma::Col<real>>& s_cache, std::list<arma::Col<real>>& y_cache, std::list<real>& rho_cache){

        std::list<real> alphas;
        arma::Col<real> q = gradient;

        auto r_s = s_cache.rbegin();
        auto r_y = y_cache.rbegin();
        auto r_rho = rho_cache.rbegin();

        while(r_s!= s_cache.rend()){
            real alpha = (*r_rho) * arma::dot(*r_s, q);
            alphas.push_front(alpha);

            q -= (*r_y)*alpha;

            ++r_s;
            ++r_y;
            ++r_rho;

        }
        arma::Col<real> direction = q * H;

        auto s = s_cache.begin();
        auto y = y_cache.begin();
        auto rho = rho_cache.begin();
        auto alpha = alphas.begin();

        while(s!=s_cache.end()){
            real beta = (*rho) * arma::dot(*y, direction);
            direction += (*s) * ((*alpha) - beta);

            ++s;
            ++y;
            ++rho;
            ++alpha;

        }

        return direction*(-1);
    }
    template<class T>
    void add_to_cache(T el,std::list<T>& cache){
        if(cache.size() == this->cache_size){
            cache.pop_front();
        }
        cache.push_back(el);
    }

};

}
}
}


#endif //NUMERICALOPTIMIZATION_L_BFGS_H
