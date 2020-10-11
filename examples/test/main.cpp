#include <string>
#include <map>
#include <iostream>
#include "functions.h"
#include "line_searches.h"
#include "methods.h"

using namespace std;
using namespace opt;

int main() {
	cout.precision(10);
    cout << fixed;

    const int n = 100;

    // typedef opt::function::ext_rosenbrock<double> func;
    // typedef opt::function::ext_himmelblau<double> func;
    // typedef opt::function::quartc<double> func;
     typedef opt::function::ext_maratos<double> func;
    // typedef opt::function::ext_tridiag2<double> func;
    // typedef opt::function::ext_white_holst<double> func;
    // typedef opt::function::diagonal4<double> func;
    // typedef opt::function::raydan1<double> func;
    // typedef opt::function::cube<double> func;
    // typedef opt::function::full_hessian2<double> func;
    // typedef opt::function::ext_hiebert<double> func;
    // typedef opt::function::part_pert_quad<double> func;
    // typedef opt::function::ext_psc1<double> func;
    // typedef opt::function::ext_quad_pen_qp1<double> func;
    // typedef opt::function::almost_pert_quad<double> func;
    // typedef opt::function::diagonal1<double> func;
    // typedef opt::function::gen_psc1<double> func;
    // typedef opt::function::fletchcr<double> func;
    // typedef opt::function::arglinb<double> func;
    // typedef opt::function::bdqrtic<double> func;
    // typedef opt::function::ext_tridiag1<double> func;
    // typedef opt::function::gen_white_holst<double> func;

    // method::gradient::gradient_descent<double> opt;
    // method::gradient::momentum<double> opt;
     method::gradient::barzilai_borwein<double> opt;
    // method::gradient::scalar_correction<double> opt;
    // method::conjugate_gradient::fletcher_reeves<double> opt;
    // method::conjugate_gradient::polak_ribiere<double> opt;
    // method::conjugate_gradient::hestenes_stiefel<double> opt;
    // method::conjugate_gradient::dai_yuan<double> opt;
    // method::conjugate_gradient::cg_descent<double> opt;
    // method::quasi_newton::sr1<double> opt;
    // method::quasi_newton::dfp<double> opt;
    // method::quasi_newton::bfgs<double> opt;
    // method::quasi_newton::l_bfgs<double> opt;
    // method::newton::newton_line_search<double> opt;
    // method::modified_newton::goldstein_price<double> opt;
    // method::modified_newton::levenberg<double> opt;
    // method::modified_newton::levenberg_marquardt<double> opt;

    // method::trust_region::dogleg<double> opt;
   //  method::trust_region::doglegSR1<double> opt;

    map<string, double> params;
    // line_search::binary<double> ls(params);
    // line_search::fixed_step_size<double> ls(params);
    // line_search::armijo<double> ls(params);
    // line_search::goldstein<double> ls(params);
    // line_search::wolfe<double> ls(params);
    // line_search::strong_wolfe<double> ls(params);
    // line_search::approx_wolfe<double> ls(params);
     line_search::nonmonotone<double> ls(params);

    opt::function::function<double> f = func::get_function();
    arma::Col<double> x = f.starting_point(n);    
   
    // arma::Col<double> x({1, 2, 3, 4, 5, 6});

     /*cout << "x:" << "\n" << x << "\n";
     cout << "func(x):" << "\n" << func::func(x) << "\n";
     cout << "grad(x):" << "\n" << func::gradient(x) << "\n";
     cout << "hess(x):" << "\n" << func::hessian(x) << "\n"; return 0;*/

    //cout << "n: " << n << "\n";
    
    /*cout << "Line search parameters:\n";
    for (auto e : params) {
        cout << e.first << ": " << e.second << "\n";
    }*/

     opt(f, ls, x);

    // opt(f, x); // for trust region methods 

    cout << "xMin: [";
    if (x.size() >= 4) {
        for (int i = 0; i < 4; i++) {
            cout << x[i] << " ";
        }
        cout << "... ]\n";
    }
    else {
        cout << "[ " << x << "]\n";
    }

    cout << "fMin: " << opt.get_f_min() << "\n";
    cout << "grNorm: " << opt.get_gr_norm() << "\n";
    cout << "iterNum: " << opt.get_iter_count() << "\n";
    cout << "cpuTime (s): " << opt.get_cpu_time() << "\n";
    cout << "funcEval: " << opt.get_f_call_count() << "\n";
    cout << "gradEval: " << opt.get_g_call_count() << "\n";
    cout << "hessEval: " << opt.get_h_call_count() << "\n";

    return 0;
}
