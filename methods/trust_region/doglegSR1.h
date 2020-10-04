#ifndef DOGLEG_SR1_H
#define DOGLEG_SR1_H

#include "../base_method.h"
#include <cmath>

namespace opt
{
	namespace method
	{
		namespace trust_region
		{
			template <class real>

			class doglegSR1 : public base_method<real>
			{
			public:

				real eta = 0.001;
				real trust_delta = 1;
				real max_delta = 1e+9;
				real r = 1e-8;

				doglegSR1() :base_method<real>() {}
				doglegSR1(real epsilon) :base_method<real>(epsilon) {}
				doglegSR1(real epsilon, size_t max_iter) :base_method<real>(epsilon, max_iter) {}
				doglegSR1(real epsilon, size_t max_iter, real working_precision) :base_method<real>(epsilon, max_iter, working_precision) {}

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

					size_t n = x.size();

					arma::Mat<real>B = arma::eye<arma::Mat<real>>(n,n);
					arma::Mat<real>H = arma::eye<arma::Mat<real>>(n,n);

					while (arma::norm(gr) > this->epsilon && this->iter_count < this->max_iter && fabs(f_prev - f_curr) / (1 + fabs(f_curr)) > this->working_precision)
					{

					
						arma::Col<real> d = dogleg_direction(gr, B, H, trust_delta);

						f_curr = f(x + d);
						m_curr = quadratic_model(f_prev, d, gr, B);

						real rho = (f_prev - f_curr) / (m_prev - m_curr);

						if (rho < 0.1)
							trust_delta = 0.5 * arma::norm(d);
						else
							if (rho > 0.75 && arma::norm(d)> 0.8*trust_delta)
								trust_delta = 2 * trust_delta < max_delta ? 2 * trust_delta : max_delta;

						arma::Col<real>gr1 = f.gradient(x+d);

						arma::Col<real> y = gr1 - gr;

						arma::Col<real>temp1 = d - H * y;
						arma::Col<real>temp2 = y - B * d;

						if (arma::dot(temp1, y) >= r * arma::norm(y) * arma::norm(temp1))
						{
							H += (temp1 * temp1.t()) / (arma::dot(temp1, y));
							B += (temp2 * temp2.t()) / (arma::dot(temp2, d));
						}


						if (rho > eta)
						{
							f_prev = f_curr;
							m_prev = f_prev;

							x = x+d;
							gr = gr1;
							
							f_curr = f_prev + 1;
							++this->iter_count;
						}


					}

					this->toc();
					this->f_min = f_prev;
					this->gr_norm = arma::norm(gr);
					this->f_call_count = f.get_call_count();
					this->g_call_count = f.get_grad_count();
					this->h_call_count = f.get_hess_count();

				}

				arma::Col<real> dogleg_direction(arma::Col<real>gr, arma::Mat<real>B, arma::Mat<real>H, real trust_delta)
				{
					real p = arma::dot(gr.t() * B, gr);

					arma::Col<real> dirC = -arma::dot(gr, gr) / p * gr;

					arma::Col<real> dirB = -H*gr;

					real tempb = arma::norm(dirB);
					real tempc = arma::norm(dirC);

					if (arma::norm(dirB) <= trust_delta)
						return dirB;


					if (arma::norm(dirC) > trust_delta)
						return -trust_delta * gr / arma::norm(gr);


					real c0 = arma::dot((dirB - dirC), (dirB - dirC));
					real c1 = 2 * arma::dot(2 * dirC - dirB, dirB - dirC);
					real c2 = arma::dot(2 * dirC - dirB, 2 * dirC - dirB) - trust_delta * trust_delta;

					real x1 = (-c1 + sqrt(c1 * c1 - 4 * c0 * c2)) / (2 * c0);
					real x2 = (-c1 - sqrt(c1 * c1 - 4 * c0 * c2)) / (2 * c0);

					real tau = x1 > x2 ? x1 : x2;

					return dirC + (tau - 1) * (dirB - dirC);

				}

				real quadratic_model(real f_value, arma::Col<real> d, arma::Col<real> gr, arma::Mat<real> hess)
				{
					return f_value + arma::dot(d, gr) + 0.5 * arma::dot(d.t() * hess, d);
				}


			};
		}
	}
}

 
#endif //DOGLEG_SR1_H

