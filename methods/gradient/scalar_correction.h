#ifndef SCALAR_CORRECTION_H
#define SCALAR_CORRECTION_H

#include "../base_method.h"

namespace opt
{
	namespace method
	{
		namespace gradient
		{
			template<class real>

			class scalar_correction: public base_method<real>
			{
			public:
				scalar_correction() :base_method<real>() {}
				scalar_correction(real epsilon): base_method<real>(epsilon){}
				scalar_correction(real epsilon, size_t max_iter) : base_method<real>(epsilon, max_iter) {}
				scalar_correction(real epsilon, size_t max_iter, real working_precision) :base_method<real>(epsilon, max_iter, working_precision) {}

				void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x)
				{
					this->iter_count = 0;
					ls.clear_f_vals();

					this->tic();

					real f_curr = f(x);
					real f_prev = f_curr + 1; // should always pass the first working precision condition

					arma::Col<real> gr = f.gradient(x);

					double gama = 1;
					while (arma::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev - f_curr) / (1 + fabs(f_curr)) > this->working_precision)
					{
						++this->iter_count;

						ls.push_f_val(f_curr);
						ls.set_current_f_val(f_curr);
						ls.set_current_g_val(gr);

						arma::Col<real>x_prev = x;
						arma::Col<real>gr_prev = gr;

						arma::Col<real> d = -1 * gama * gr;
						x += d * ls(f, x, d);

						f_prev = f_curr;
						f_curr = ls.get_current_f_val();
						gr = ls.get_current_g_val();

						arma::Col<real>s = x - x_prev;
						arma::Col<real>y = gr - gr_prev;
						arma::Col<real>r = s - gama*y;

						if (arma::dot(y, r) > 0)
							gama = arma::dot(s, r) / arma::dot(y, r);
						else
							gama = arma::norm(s) / arma::norm(y);

						if (gama < 0) gama = 1;

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
	}
}
#endif //SCALAR_CORRECTION_H
