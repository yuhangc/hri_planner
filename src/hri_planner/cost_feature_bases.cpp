//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/7/2017
// Last revision: 3/8/2017
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
        double xt = x(t*nX) / a;
        double yt = x(t*nX+1) / b;
        cost += std::exp(xt * xt + yt * yt);
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
        double xt = x(st) / a;
        double yt = x(st+1) / b;
        double c = std::exp(xt * xt + yt * yt);

        grad(st) = -2.0 * xt * c / a;
        grad(st+1) = -2.0 * yt * c / b;
    }
}

//----------------------------------------------------------------------------------
void GaussianCost::hessian(const Eigen::VectorXd &x, const int nX, const int T, const double a, const double b,
                           Eigen::MatrixXd &hess)
{
    hess.setZero(T*nX, T*nX);

    for (int t = 0; t < T; ++t) {
        int st = t * nX;
        double xt = x(st) / a;
        double yt = x(st+1) / b;
        double c = std::exp(xt * xt + yt * yt);

        hess(st, st) = (4.0 * xt * xt - 2.0) / (a * a);
        hess(st, st+1) = 4.0 * xt * yt / (a * b);
        hess(st+1, st) = hess(st, st+1);
        hess(st+1, st+1) = (4.0 * yt * yt - 2.0) / (b * b);
    }
}

}
