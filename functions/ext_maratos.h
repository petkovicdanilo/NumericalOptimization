#ifndef MARATOS_FUNCTION
#define MARATOS_FUNCTION

#include "function.h";

namespace opt
{
	namespace function
	{
		template <class real>

		class ext_maratos
		{
		public:

			static  const int c = 100;

			static real func(const arma::Col<real>& v)
			{
				if (v.size() <= 0 || v.size() % 2)
					throw "maratos: n must be even and positive";

				size_t n = v.size();
				real z = 0;

				for (size_t i = 0; i < n/2; i++)
				{
					real t = v[2 * i] * v[2 * i] + v[2 * i + 1] * v[2 * i + 1] - 1;
					z += v[2*i] + c * t * t;
				}
				return z;
			}

			static arma::Col<real> gradient(const arma::Col<real>& v)
			{
				size_t n = v.size();
				if (n <= 0 || n % 2)
					throw "maratos: n must be even and positive";

				arma::Col z = arma::zeros<arma::Col<real>>(n);

				for (size_t i = 0; i < n; i += 2)
				{
					z[i] = 1 + c * (4 * v[i] * v[i] * v[i] + 4 * v[i + 1] * v[i + 1] * v[i] - 4 * v[i]);
					z[i+1] = c * (4 * v[i+1] * v[i+1] * v[i+1] - 4 * v[i+1] + 4 * v[i] * v[i] * v[i+1]);
				}
				return z;
			}

			static arma::Mat<real> hessian(const arma::Col<real>& v)
			{
				size_t n = v.size();
				if (n <= 0 || n % 2)
					throw "maratos: n must be even and positive";

				arma::Mat<real>z = arma::zeros<arma::Mat<real>>(n, n);
				
				for (size_t i = 0; i < n; i += 2)
				{
					z(i,i) = c*(12*v[i]*v[i]+4*v[i+1]*v[i+1]-4);
					z(i,i+1)= c*8*v[i]*v[i+1];
					z(i + 1, i) = c * 8 * v[i] * v[i+1];
					z(i + 1, i + 1) = c * (12 * v[i + 1] * v[i + 1] - 4 + 4 * v[i] * v[i]);
				}

				return z;
			}

			static arma::Col<real> starting_point(const size_t n)
			{
				if (n <= 0 || n % 2) throw "ext_maratos_starting_point: n must be even and positive";
				arma::Col<real> z(n);
				for (size_t i = 0; i < n; i += 2)
				{
					z[i] = 1.1;
					z[i + 1] = 0.1;
				}

				return z;
			}

			static function<real> get_function() {
				return function<real>(func, gradient, hessian, starting_point);
			}

		};
	}
}
#endif //MARATOS_FUNCTION
