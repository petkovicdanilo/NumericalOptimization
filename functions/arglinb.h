#ifndef NUMERICALOPTIMIZATION_ARG_LIN_B
#define NUMERICALOPTIMIZATION_ARG_LIN_B

#include "function.h"

namespace opt
{
namespace function
{
    template <class real>
    class arglinb {
    public:
        static real func(const arma::Col<real> &v) {
            size_t n = v.size();

            if (n == 0) {
                throw "arg_lin_b: n must be positive";
            }

            real val = 0.0;

            for (size_t i = 0; i < n; i++) {
                real inner_sum = get_inner_sum(i, v);
                val += inner_sum * inner_sum;
            }

            return val;
        }

        static arma::Col<real> gradient(const arma::Col<real> &v) {
            size_t n = v.size();

            if (n == 0) {
                throw "arg_lin_b: n must be positive";
            }

            arma::Col<real> grad = arma::zeros<arma::Col<real>>(n);

            for (size_t i = 0; i < n; i++) {
                real inner_sum = get_inner_sum(i, v);

                for (size_t j = 0; j < n; j++) {
                    grad[j] += 2 * (i + 1) * (j + 1) * inner_sum;
                }
            }

            return grad;
        }

        static arma::Mat<real> hessian(const arma::Col<real> &v) {
            size_t n = v.size();

            if (n == 0) {
                throw "arg_lin_b: n must be positive";
            }

            arma::Mat<real> hes = arma::zeros<arma::Mat<real>>(n, n);

            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    for (size_t k = 0; k < n; k++) {
                        hes(j, k) += 2 * (i + 1) * (i + 1) * (j + 1) * (k + 1);
                    }
                }
            }

            return hes;
        }

        static arma::Col<real> starting_point(const size_t n) {
            return arma::ones<arma::Col<real>>(n);
        }

        static function<real> get_function() {
            return function<real>(func, gradient, hessian, starting_point);
        }

    private:
        static real get_inner_sum(size_t i, const arma::Col<real> &v) {
            real inner_sum = 0.0;
            size_t n = v.size();

            for (size_t j = 0; j < n; j++) {
                inner_sum += (i + 1) * (j + 1) * v[j] - 1;
            }

            return inner_sum;
        }
    };
} // namespace function
} // namespace opt

#endif //NUMERICALOPTIMIZATION_ARG_LIN_B
