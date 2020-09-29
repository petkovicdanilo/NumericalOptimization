#ifndef EXTENDED_TRIDIAGONAL2_FUNCTION
#define EXTENDED_TRIDIAGONAL2_FUNCTION

#include "function.h";

namespace opt
{
	namespace function
	{
		template <class real>

	

		class ext_tridiag2
		{
		public:
		    inline static const double c=0.1;

			static real func(const arma::Col<real>& v)
			{
				if (v.size() <= 0)
					throw "ext_tridiagonal2_func: n must be positive";

				real z = 0;
				size_t n = v.size();

				for (size_t i = 0; i < n-1; i++)
				{
					z += (v[i] * v[i + 1] - 1) * (v[i] * v[i + 1] - 1) + c * (v[i]+1)*(v[i+1]+1);
				}

				return z;
			}

			static arma::Col<real> gradient(const arma::Col<real>& v)
			{
				if (v.size() <= 0)
					throw "ext_tridiagonal2_gradient: n must be positive";

				size_t n = v.size();
				arma::Col<real> z(n);
				
				z[0] = 2 * v[0] * v[1] * v[1] - 2 * v[1] + c * v[2] + c;
				z[n - 1] = 2*v[n-2]*v[n-2]*v[n-1]-2*v[n-2]+c*v[n-2]+c;
				
				for (size_t i = 1; i < n-1; i++)
				{
					z[i] = 2 * v[i] * (v[i - 1] * v[i - 1] + v[i + 1] * v[i + 1]) - 2 * (v[i - 1] + v[i + 1]) + c * (v[i - 1] + v[i + 1]) + 2 * c;
				}

				return z;
			}

			static arma::Mat<real> hessian(const arma::Col<real>& v)
			{
				if (v.size() <= 0)
					throw "ext_tridiagonal2_hessian: n must be positive";

				size_t n = v.size();
				arma::Mat<real>z = arma::zeros<arma::Mat<real>>(n, n);

				z(0, 0) = 2 * v[1] * v[1];
				z(0, 1) = 4 * v[0] * v[1] + c - 2;
				z(n - 1, n - 2) = 4 * v[n - 2] * v[n - 1]-2+c;
				z(n - 1, n - 1) = 2 * v[n - 2] * v[n - 2];

				for (size_t i = 1; i < n-1; i ++)
				{
					z(i, i - 1) = 4*v[i]*v[i-1]-2+c;
					z(i, i) = 2*(v[i-1]*v[i-1]+v[i+1]*v[i+1]);
					z(i, i + 1) = 4 * v[i] * v[i + 1] - 2 + c;
				}

				return z;
			}

			static arma::Col<real> starting_point(const size_t n)
			{
				if (n <= 0)
					throw "ext_tridiagonal2_starting_point: n must be positive";

				return arma::ones<arma::Col<real>>(n);
			}

			static function<real> get_function() {
				return function<real>(func, gradient, hessian, starting_point);
			}
			
		};
	}
}

#endif 