//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/7/2017
// Last revision: 3/7/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <cmath>

#include "hri_planner/dynamics.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
ConstAccDynamics::ConstAccDynamics(double dt): DynamicsBase(4, 2, dt) {
    // create the linear dynamics matrices A, B
    A_.setIdentity(nX_, nX_);
    A_(0, 2) = dt;
    A_(1, 3) = dt;

    B_.setZero(nX_, nU_);
    B_(0, 0) = 0.5 * dt_ * dt_;
    B_(1, 1) = 0.5 * dt_ * dt_;
    B_(2, 0) = dt_;
    B_(3, 1) = dt_;
}

//----------------------------------------------------------------------------------
void ConstAccDynamics::forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new)
{
    x_new = A_ * x + B_ * u;
}

//----------------------------------------------------------------------------------
void ConstAccDynamics::grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx)
{
    Jx = A_;
}

//----------------------------------------------------------------------------------
void ConstAccDynamics::grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju)
{
    Ju = B_;
}

//----------------------------------------------------------------------------------
void DifferentialDynamics::forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new)
{
    x_new.setZero();

    const double th = x(2);
    if (u(1) < om_tol_) {
        // approximately linear motion only
        x_new(0) = x(0) + u(0) * std::cos(th) * dt_;
        x_new(1) = x(1) + u(0) * std::sin(th) * dt_;
        x_new(2) = x(2) + u(1) * dt_;
    }
    else {
        // constant curvature motion
        const double R = u(0) / u(1);
        const double dth = u(1) * dt_;

        x_new(0) = x(0) + R * (-std::sin(th) + std::sin(th + dth));
        x_new(1) = x(1) + R * (std::cos(th) + std::cos(th + dth));
        x_new(2) = x(2) + dth;
    }
}

//----------------------------------------------------------------------------------
void DifferentialDynamics::grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx)
{
    Jx.setIdentity();

    const double th = x(2);
    if (u(1) < om_tol_) {
        Jx(0, 2) = u(0) * dt_ * (-std::sin(th));
        Jx(1, 2) = u(0) * dt_ * std::cos(th);
    }
    else {
        const double R = u(0) / u(1);
        const double th_new = th + u(1) * dt_;

        Jx(0, 2) = R * (-std::cos(th) + std::cos(th_new));
        Jx(1, 2) = R * (std::sin(th) + std::sin(th_new));
    }
}

//----------------------------------------------------------------------------------
void DifferentialDynamics::grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju)
{
    Ju.setZero();

    const double th = x(2);
    if (u(1) < om_tol_) {
        Ju(0, 0) = dt_ * std::cos(th);
        Ju(1, 0) = dt_ * std::sin(th);
        Ju(2, 1) = dt_;
    }
    else {
        double om_inv = 1.0 / u(1);
        double th_new = th + u(1) * dt_;
        double R = u(0) / u(1);

        Ju(0, 0) = om_inv * (-std::sin(th) + std::sin(th_new));
        Ju(1, 0) = om_inv * (std::cos(th) - std::cos(th_new));
        Ju(0, 1) = R * (-om_inv * std::sin(th_new) + dt_ * std::cos(th_new));
        Ju(1, 1) = R * (om_inv * std::cos(th_new) + dt_ * std::sin(th_new));
        Ju(2, 1) = dt_;
    }
}

} // namespace
