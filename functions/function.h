#ifndef PROJEKATC___FUNCTION_H
#define PROJEKATC___FUNCTION_H

#include <armadillo>

namespace opt {
namespace function {

template<class real>
class function {
public:
	using func = real(*)(const arma::Col<real>&);
	using grad = arma::Col<real>(*)(const arma::Col<real>&);
	using hess = arma::Mat<real>(*)(const arma::Col<real>&);
	using start = arma::Col<real>(*)(const size_t);

	function(func f, grad g, hess h, start s) : f(f), g(g), h(h), s(s), call_count(0), grad_count(0), hess_count(0) {}

	real operator()(const arma::Col<real>& x) {
		++call_count;
		return f(x);
	}

	arma::Col<real> gradient(const arma::Col<real>& x) {
		++grad_count;
		return g(x);
	}

	arma::Mat<real> hessian(const arma::Col<real>& x) {
		++hess_count;
		return h(x);
	}

	arma::Col<real> starting_point(const size_t n) {
		return s(n);
	}

	size_t get_call_count() const {
		return call_count;
	}

	size_t get_grad_count() const {
		return grad_count;
	}

	size_t get_hess_count() const {
		return hess_count;
	}
protected:
	func f;
	grad g;
	hess h;
	start s;
	size_t call_count;
	size_t grad_count;
	size_t hess_count;
};

}
}

#endif //PROJEKATC___FUNCTION_H
