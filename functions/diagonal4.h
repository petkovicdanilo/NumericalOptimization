#ifndef DIAGONAL4_FUNCTION
#define DIAGONAL4_FUNCTION

#include "function.h";

namespace opt
{
	namespace function
	{
		template<class real>

		class diagonal4
		{
		public:

			static const int c = 100;

			static real func(const arma::Col<real>& v)
			{
				size_t n = v.size();

				if (n <= 0 || n % 2)
					throw "diagonal4: n must be even and positive";

				real z = 0;
				for (size_t i = 0; i < n / 2; i++)
				{
					z += (v[2 * i] * v[2 * i] + c * v[2 * i + 1] * v[2 * i + 1]);
				}
				z /= 2;
				return z;
			}

			static arma::Col<real> gradient(const arma::Col<real>& v)
			{
				size_t n = v.size();
				if (n <= 0 || n % 2)
					throw"diagonal4: n must be even and positive";

				arma::Col<real>z(n);

				for (size_t i = 0; i < n; i += 2)
				{
					z[i] = v[i];
					z[i + 1] = c * v[i + 1];
				}

				return z;
			}

			static arma::Mat<real> hessian(const arma::Col<real>& v)
			{
				size_t n = v.size();
				if (n <= 0 || n % 2)
					throw "diagonal4: n must be even and positive";

				arma::Mat<real>z = arma::zeros<arma::Mat<real>>(n,n);
				
				for (size_t i = 0; i < n; i += 2)
				{
					z(i,i) = 1;
					z(i,i+1) = 0;
				}

				for (size_t i = 1; i < n; i += 2)
				{
					z(i,i-1) = 0;
					z(i,i) = c;
				}
				return z;
			}

			static arma::Col<real> starting_point(const size_t n)
			{
				if (n <= 0 || n % 2)
					throw "diagonal4_starting_point: n must be even and positive";
				return arma::ones<arma::Col<real>>(n);
			}

			static function<real> get_function() {
				return function<real>(func, gradient, hessian, starting_point);
			}

		};
	}
}
#endif //DIAGONAL4_FUNCTION
