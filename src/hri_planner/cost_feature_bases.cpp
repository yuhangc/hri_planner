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
float GaussianCost::compute(const Eigen::VectorXf &x, const int nX, const int T, const float a, const float b)
{
    float cost = 0.0f;

    for (int t = 0; t < T; ++t) {
        float xt = x(t*nX) / a;
        float yt = x(t*nX+1) / b;
        cost += std::exp(xt * xt + yt * yt);
    }

    return cost;
}

//----------------------------------------------------------------------------------
void GaussianCost::grad(const Eigen::VectorXf &x, const int nX, const int T, const float a, const float b,
                        Eigen::VectorXf &grad)
{
    grad.setZero(T * nX);

    for (int t = 0; t < T; ++t) {
        int st = t * nX;
        float xt = x(st) / a;
        float yt = x(st+1) / b;
        float c = std::exp(xt * xt + yt * yt);

        grad(st) = -2.0f * xt * c / a;
        grad(st+1) = -2.0f * yt * c / b;
    }
}

//----------------------------------------------------------------------------------
void GaussianCost::hessian(const Eigen::VectorXf &x, const int nX, const int T, const float a, const float b,
                           Eigen::MatrixXf &hess)
{
    hess.setZero(T*nX, T*nX);

    for (int t = 0; t < T; ++t) {
        int st = t * nX;
        float xt = x(st) / a;
        float yt = x(st+1) / b;
        float c = std::exp(xt * xt + yt * yt);

        hess(st, st) = (4.0f * xt * xt - 2.0f) / (a * a);
        hess(st, st+1) = 4.0f * xt * yt / (a * b);
        hess(st+1, st) = hess(st, st+1);
        hess(st+1, st+1) = (4.0f * yt * yt - 2.0f) / (b * b);
    }
}

}
