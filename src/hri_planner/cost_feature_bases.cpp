//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/7/2018
// Last revision: 3/13/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <cmath>

#include "hri_planner/cost_feature_bases.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
double GaussianCost::compute(const Eigen::VectorXd &x, const int nX, const int T, const double a, const double b)
{
    double cost = 0.0;

    for (int t = 0; t < T; ++t) {
        double xt = x(t*2) / a;
        double yt = x(t*2+1) / b;
        cost += std::exp(-(xt * xt + yt * yt));
    }

    return cost;
}

//----------------------------------------------------------------------------------
void GaussianCost::grad(const Eigen::VectorXd &x, const int nX, const int T, const double a, const double b,
                        Eigen::VectorXd &grad)
{
    grad.setZero(T * nX);

    for (int t = 0; t < T; ++t) {
        int st = t * nX;
        double xt = x(t*2) / a;
        double yt = x(t*2+1) / b;
        double c = std::exp(-(xt * xt + yt * yt));

        grad(st) = -2.0 * xt * c / a;
        grad(st+1) = -2.0 * yt * c / b;
    }
}

//----------------------------------------------------------------------------------
void GaussianCost::hessian(const Eigen::VectorXd &x, const int nX1, const int nX2, const int T,
                           const double a, const double b, Eigen::MatrixXd &hess)
{
    hess.setZero(T*nX1, T*nX2);

    for (int t = 0; t < T; ++t) {
        int st1 = t * nX1;
        int st2 = t * nX2;
        double xt = x(t*2) / a;
        double yt = x(t*2+1) / b;
        double c = std::exp(-(xt * xt + yt * yt));

        hess(st1, st2) = (4.0 * xt * xt - 2.0) * c / (a * a);
        hess(st1, st2+1) = 4.0 * xt * yt * c / (a * b);
        hess(st1+1, st2) = hess(st1, st2+1);
        hess(st1+1, st2+1) = (4.0 * yt * yt - 2.0) * c / (b * b);
    }
}

}
