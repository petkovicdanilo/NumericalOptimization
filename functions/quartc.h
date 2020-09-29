#ifndef QUARTC_FUNCTION
#define QUARTC_FUNCTION

#include "function.h"

namespace opt 
{
	namespace function
	{
		template<class real>
		class quartc
		{
		public:
			static real func(const arma::Col<real>& v)
			{
				if (v.size() <= 0)
					throw "quartc_func: n must be positive";

				size_t n = v.size();
				real z = 0;

				for (size_t i = 0; i < n; i++)
				{
					real t = v[i] - 1;
					z += t * t * t * t;
				}
				return z;
			}

			static arma::Col<real> gradient(const arma::Col<real>& v)
			{
				if (v.size() <= 0)
					throw "quartc_gradient: n must be positive";

				size_t n = v.size();
				arma::Col<real> z = arma::zeros<arma::Col<real>>(n);

				for (size_t i = 0; i < n; i++)
				{
					z[i] = 4 * v[i] * v[i] * v[i] - 12 * v[i] * v[i] + 12 * v[i] - 4;
				}

				return z;
			}

			static arma::Mat<real> hessian(const arma::Col<real>& v)
			{
				if (v.size() <= 0)
					throw "quartc_hessian: n must be positive";

				size_t n = v.size();
				arma::Mat<real> z = arma::zeros<arma::Mat<real>>(n,n);

				for (size_t i = 0; i < n; i++)
				{
					z(i, i) = 12 * v[i] * v[i] - 24 * v[i] + 12;
				}
				return z;
			}

			static arma::Col<real> starting_point(const size_t n)
			{
				if (n <= 0)throw "quartc_starting_point:n mus be positive";
				arma::Col<real>z(n);

				for (size_t i = 0; i < n; i++)
					z[i] = 2;

				return z;
			}

			static function<real> get_function() {
				return function<real>(func, gradient, hessian, starting_point);
			}

		};
	}
}
#endif
