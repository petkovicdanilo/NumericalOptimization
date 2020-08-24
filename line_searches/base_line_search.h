#ifndef PROJEKATC___LINE_SEARCH_H
#define PROJEKATC___LINE_SEARCH_H

#include <string>
#include <map>
#include <cmath>
#include <vector>
#include <armadillo>
#include "../functions/function.h"

namespace opt {
namespace line_search {

template<class real>
class base_line_search {
public:
    base_line_search() : iter_count(0), c(0.0) {}

    virtual real operator()(function::function<real>& f, arma::Col<real>& x, arma::Col<real>& d) = 0;

    size_t get_iter_count() const {
        return iter_count;
    }

    real get_current_f_val() const {
        return current_f_val;
    }

    void set_current_f_val(real f_val) {
        current_f_val = f_val;
    }

    arma::Col<real> get_current_g_val() const {
        return current_g_val;
    }

    void set_current_g_val(arma::Col<real> g_val) {
        current_g_val = g_val;
    }

    void set_c(real c) {
        this->c = c;
    }

    void push_f_val(real val) {
        f_values.push_back(val);
    }

    void clear_f_vals() {
        f_values.clear();
    }
protected:
    size_t iter_count; // number of iterations in the line search (inner) loop
    std::vector<real> f_values; // list of all function values from previous iterations
    real current_f_val; // the current operating function value of x; used to exchange data between the method and the line search
    arma::Col<real> current_g_val; // the current operating gradient value of x; used to exchange data between the method and the line search
    real c; // number based on f_curr only used in approx_wolfe line search passed from cg_descent method

    // Copies pairs from the second to the first map. Used in concrete constructors to override
    // default line search parameters.
    void rest(std::map<std::string, real>& params, std::map<std::string, real>& custom_params) {
        for (auto e : custom_params) {
            params[e.first] = e.second;
        }
    }

    // Computes point t between points t1 and t2 by cubic interpolation.
    real cubic_interpolation(real t1, real t2, real val1, real val2, real der1, real der2) {
        real d1 = der1 + der2 - 3*(val1-val2)/(t1-t2);
        real d2 = sqrt(d1*d1 - der1*der2);
        real tmp = t2 - (t2-t1)*(der2+d2-d1)/(der2-der1+2*d2);
        real t;

        if (tmp >= 0) {
            d2 = sqrt(d1*d1 - der1*der2);
            t = t2 - (t2-t1)*(der2+d2-d1)/(der2-der1+2*d2);
        } else {
            t = t1 - 1;
        }

        // if minimum is is not in the interval (t1, t2) then minimum is in t1
        if (t < t1 || t > t2) {
            t = t1;
        }

        return t;
    }

    // Computes the line search initial step according to the idea proposed by
    // Nocedal and Wright "Numerical Optimization",
    // lecture "The Initial Step Length", chapter 3, page 58.
    // Must be used starting from the second iteration of the outer loop as
    // f_prev is required.
    real compute_initial_step(real f_curr, real f_prev, arma::Col<real> grad, arma::Col<real> d) {
        return fmin(1.0, 1.01*fabs(2*(f_curr-f_prev)/arma::dot(d, grad)));
    }
};

}
}

#endif //PROJEKATC___LINE_SEARCH_H
