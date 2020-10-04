#ifndef PROJEKATC___METHODS_H
#define PROJEKATC___METHODS_H

#include "methods/gradient/gradient_descent.h"
#include "methods/gradient/momentum.h"
#include "methods/gradient/barzilai_borwein.h"
#include "methods/gradient/scalar_correction.h"
#include "methods/conjugate_gradient/fletcher_reeves.h"
#include "methods/conjugate_gradient/polak_ribiere.h"
#include "methods/conjugate_gradient/hestenes_stiefel.h"
#include "methods/conjugate_gradient/dai_yuan.h"
#include "methods/conjugate_gradient/cg_descent.h"
#include "methods/quasi_newton/sr1.h"
#include "methods/quasi_newton/dfp.h"
#include "methods/quasi_newton/bfgs.h"
#include "methods/quasi_newton/l_bfgs.h"
#include "methods/newton/newton_line_search.h"
#include "methods/modified_newton/goldstein_price.h"
#include "methods/modified_newton/levenberg.h"
#include "methods/modified_newton/levenberg_marquardt.h"
#include "methods/trust_region/dogleg.h"
#include "methods/trust_region/doglegSR1.h"

#endif //PROJEKATC___METHODS_H
