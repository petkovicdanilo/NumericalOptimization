#ifndef DOGLEG_H
#define DOGLEG_H

#include "../base_method.h"
#include <cmath>

namespace opt
{
	namespace method
	{
		namespace trust_region
		{
			template <class real>

			class dogleg : public base_method<real>
			{
			public:

				real eta = 0.001;
				real trust_delta = 5;
				real max_delta = 1e+9;

				dogleg() :base_method<real>() {}
				dogleg(real epsilon) :base_method<real>(epsilon) {}
				dogleg(real epsilon, size_t max_iter) :base_method<real>(epsilon, max_iter) {}
				dogleg(real epsilon, size_t max_iter, real working_precision) :base_method<real>(epsilon, max_iter, working_precision) {}

				void operator()(function::function<real>& f, line_search::base_line_search<real>& ls, arma::Col<real>& x) {}

				void operator()(function::function<real>& f, arma::Col<real>& x)
				{
					this->iter_count = 0;

					this->tic();

					real f_prev = f(x);
					real f_curr = f_prev + 1;
					real m_prev = f_prev;
					real m_curr;

					arma::Col<real>gr = f.gradient(x);
					arma::Mat<real>hess = f.hessian(x);

					while (arma::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev - f_curr) / (1 + fabs(f_curr)) > this->working_precision)
					{
						
						arma::Col<real> d = dogleg_direction(gr, hess, trust_delta);
						real normd = arma::norm(d);
						
						f_curr = f(x + d);
						m_curr = quadratic_model(f_prev, d, gr, hess);
						
						real rho = (f_prev - f_curr) / (m_prev - m_curr);
					
						if (rho < 0.1)
							trust_delta = 0.25 * arma::norm(d);
						else
							if (rho > 0.75 && fabs(arma::norm(d) - trust_delta) < this->epsilon)
								trust_delta = 2 * trust_delta < max_delta ? 2 * trust_delta : max_delta;

						if (rho > eta)
						{
							f_prev = f_curr;
							m_prev = f_prev;

							x = x + d;
							
							f_curr = f_prev + 1;
							++this->iter_count;
						}

						gr = f.gradient(x);
						hess = f.hessian(x);

					}

					this->toc();
					this->f_min = f_prev;
					this->gr_norm = arma::norm(gr);
					this->f_call_count = f.get_call_count();
					this->g_call_count = f.get_grad_count();
					this->h_call_count = f.get_hess_count();

				}

				arma::Col<real> dogleg_direction(arma::Col<real>gr, arma::Mat<real>hess, real trust_delta)
				{
					real p = arma::dot(gr.t() * hess, gr);

					arma::Col<real> dirC = -arma::dot(gr,gr)/ p *gr;

					arma::Col<real> dirB = -arma::solve(hess, gr);
					

					if (arma::norm(dirB) <= trust_delta)
						return dirB;
				

					if (arma::norm(dirC) > trust_delta)
						return -trust_delta * gr / arma::norm(gr);


					real c0 = arma::dot((dirB - dirC), (dirB - dirC));
					real c1 = 2 * arma::dot( 2*dirC-dirB, dirB - dirC);
					real c2 = arma::dot(2*dirC-dirB,2*dirC-dirB) -trust_delta * trust_delta;

					real x1 = (-c1 + sqrt(c1 * c1 - 4 * c0 * c2)) / (2 * c0);
					real x2 = (-c1 - sqrt(c1 * c1 - 4 * c0 * c2)) / (2 * c0);

					real tau = x1 > x2 ? x1 : x2;

					return dirC + (tau - 1) * (dirB - dirC);
					
				}

				real quadratic_model(real f_value, arma::Col<real> d, arma::Col<real> gr, arma::Mat<real> hess)
				{
					return f_value + arma::dot(d, gr) + 0.5 * arma::dot(d.t()*hess,d);
				}
			

			};
		}
	}
}


#endif //DOGLEG_H