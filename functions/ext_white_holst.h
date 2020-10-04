#ifndef EXT_WHITE_HOLST_FUNCTION
#define EXT_WHITE_HOLST_FUNCTION

#include "function.h";

namespace opt
{
	namespace function
	{
		template <class real>

		class ext_white_holst
		{
		public:

			static const int c = 100;

			static real func(const arma::Col<real>& v)
			{
				size_t n = v.size();

				if (n <= 0 || n % 2)
					throw "ext_white_holst_func: n must be even and positive";

				real z = 0;

				for (size_t i = 0; i < n ; i+=2)
				{
					real t = v[i] * v[i] * v[i];
					z += c * v[i + 1]*v[i+1] + c *t*t  - 2 *c*v[i+1]*t + 1 + v[i]*v[i] - 2*v[i];
				}

				return z;
			}

			static arma::Col<real> gradient(const arma::Col<real>& v)
			{
				size_t n = v.size();
				if (n <= 0 || n % 2)
					throw "ext_white_host_gradient: n must be even and positive";

				arma::Col<real> z(n);
				size_t t = 6 * c;
				for (size_t i = 0; i <n; i+=2)
				{
					real p = v[i] * v[i];
					z[i] = t*p*p*v[i]-t*p*v[i+1]+2*v[i]-2;
					z[i + 1] = 2*c*(v[i+1]-p*v[i]);
				}
				return z;
			}

			static arma::Mat<real> hessian(const arma::Col<real>& v)
			{
				size_t n = v.size();

				if (n <= 0 || n % 2)
					throw "ext_white_holst_hessian: n must be even and positive";

				arma::Mat<real>z = arma::zeros<arma::Mat<real>>(n, n);
				for (int i = 0; i < n; i+=2)
				{
					real p = v[i] * v[i];
					z(i, i) =30*c*p*p-12*c*v[i]*v[i+1]+2;
					z(i, i + 1) = -6*c*p;
					z(i + 1, i) = z(i,i+1);
					z(i + 1, i + 1) = 2*c;
				}
				return z;
			}

			static arma::Col <real> starting_point(const size_t n)
			{
				if (n <= 0 || n % 2)
					throw "ext_white_holst: n must be even and positive";
				
				arma::Col<real>z(n);

				for (size_t i = 0; i < n; i += 2)
				{
					z[i] = -1.2;
					z[i + 1] = 1;
				}
				return z;
			}

			static function<real> get_function() {
				return function<real>(func, gradient, hessian, starting_point);
			}
		};
	}
}
#endif
