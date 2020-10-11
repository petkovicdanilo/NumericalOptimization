#ifndef NONMONOTONE_H
#define NONMONOTONE_H

#include "base_line_search.h" 
#include<vector>

namespace opt
{
	namespace line_search
	{
		template <class real>

		class nonmonotone : public base_line_search<real>
		{
		private:
			real steepness;
			real initial_step;
			int M;
		public:

			nonmonotone(std::map<std::string, real>& params)
			{
				std::map<std::string, real> p;
				p["steepness"] = 1e-4;
				p["initial_step"] = 1;
				p["M"] = 10;
				this->rest(p, params);
				steepness = p["steepness"];
				initial_step = p["initial_step"];
				M = p["M"];
				params = p;
			}

			real operator()(function::function<real>& f, arma::Col<real>& x, arma::Col<real>& d)
			{
				
				this->iter_count = 0;
				real f0 = this->current_f_val;

				real pad = arma::dot(this->current_g_val, d);
				real a_curr = this->f_values.size() >= 2 ? this->compute_initial_step(this->f_values.end()[-1], this->f_values.end()[-2], this->current_g_val, d) : initial_step;
				
				real f_curr, f_prev, a_prev;
				f_curr = f(x + d * a_curr);
			
				
				vector<real> curr_values;
				curr_values.push_back(f0);
				curr_values.push_back(f_curr);

				real maxval = f_curr > f0 ? f_curr : f0;
				
				while (f_curr > maxval + steepness * pad * a_curr)
				{
					++this->iter_count;
					real a_new;

					if (this->iter_count == 1)
					{
						a_new = pad * a_curr * a_curr / 2 / (f0 - f_curr + pad * a_curr);
					}
					else
					{
						real cubic = a_prev * a_prev * (f_curr - f0);
						cubic -= a_curr * a_prev * a_prev * pad;
						cubic += a_curr * a_curr * (f0 - f_prev + a_prev * pad);
						cubic /= a_curr * a_curr * (a_curr - a_prev) * a_prev * a_prev;

						real quadr = -cubic * a_curr * a_curr * a_curr - f0 + f_curr - a_curr * pad;
						quadr /= a_curr * a_curr;

						a_new = (-quadr + sqrt(quadr * quadr - 3 * cubic * pad)) / (3 * cubic);
					}
					a_prev = a_curr;
					a_curr = a_new;

					f_prev = f_curr;
					f_curr = f(x + a_new * d);
					curr_values.push_back(f_curr);

					// calculating new maximum of last M iterations and erasing first value if needed	
					if (curr_values.size() > M) // f_values should not contain values from more then last M iterations
					{
						curr_values.erase(f_values.begin());

						maxval = curr_values[0];
						for (size_t i = 1; i < curr_values.size(); i++)
							if (curr_values[i] > maxval)maxval = curr_values[i];
					}
					else
					{
						if (curr_values.end()[0] > maxval)maxval = curr_values.end()[0];
					}

					
				}

				this->current_f_val = f_curr;
				this->current_g_val = f.gradient(x + d * a_curr);
				return a_curr;
			}

			
		};
	}
}

#endif //NONMONOTONE_H
