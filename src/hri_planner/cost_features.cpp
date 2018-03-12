//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/8/2017
// Last revision: 3/8/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include "hri_planner/cost_features.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
double HumanVelCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    double cost = 0.0;

    int nX = human_traj.state_size();
    for (int t = 0; t < human_traj.horizon(); ++t) {
        double vx = human_traj.x(t*nX+2);
        double vy = human_traj.x(t*nX+3);
        cost += vx * vx + vy * vy;
    }

    return cost;
}

//----------------------------------------------------------------------------------
void HumanVelCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    // compute the gradient w.r.t. xh first
    Eigen::VectorXd grad_x(human_traj.traj_state_size());

    int nX = human_traj.state_size();
    for (int t = 0; t < human_traj.horizon(); ++t) {
        int xs = t * nX;
        grad_x(xs) = 0;
        grad_x(xs+1) = 0;
        grad_x(xs+2) = 2.0 * human_traj.x(xs+2);
        grad_x(xs+3) = 2.0 * human_traj.x(xs+3);
    }

    // compute gradient w.r.t. uh
    grad = human_traj.Ju.transpose() * grad_x;
}

//----------------------------------------------------------------------------------
void HumanVelCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    // compute the hessian w.r.t. xh first
    Eigen::MatrixXd hess_x;
    hess_x.setZero(human_traj.traj_state_size(), human_traj.traj_state_size());

    int nX = human_traj.state_size();
    for (int t = 0; t < human_traj.horizon(); ++t) {
        int xs = t * nX;
        hess_x(xs+2, xs+2) = 2.0;
        hess_x(xs+3, xs+3) = 2.0;
    }

    // compute hessian w.r.t. uh
    hess = human_traj.Ju.transpose() * hess_x * human_traj.Ju;
}

//----------------------------------------------------------------------------------
double HumanAccCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    return human_traj.u.squaredNorm();
}

//----------------------------------------------------------------------------------
void HumanAccCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    grad = 2.0 * human_traj.u;
}

//----------------------------------------------------------------------------------
void HumanAccCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    hess.setIdentity();
    hess *= 2.0;
}

//----------------------------------------------------------------------------------
double HumanGoalCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    int xs = human_traj.traj_state_size() - human_traj.state_size();
    double x_diff = x_goal_(0) - human_traj.x(xs);
    double y_diff = x_goal_(1) - human_traj.x(xs+1);

    double cost = std::sqrt(x_diff * x_diff + y_diff * y_diff);
    return cost;
}

//----------------------------------------------------------------------------------
void HumanGoalCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    Eigen::Vector2d grad_x;
    int xs = human_traj.traj_state_size() - human_traj.state_size();
    double x_diff = x_goal_(0) - human_traj.x(xs);
    double y_diff = x_goal_(1) - human_traj.x(xs+1);
    double d = std::sqrt(x_diff * x_diff + y_diff * y_diff) + reg_;

    grad(0) = x_diff / reg_;
    grad(1) = y_diff / reg_;

    grad = human_traj.Ju.middleRows(xs, 2).transpose() * grad_x;
}

//----------------------------------------------------------------------------------
void HumanGoalCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    Eigen::Vector2d hess_x;
    int xs = human_traj.traj_state_size() - human_traj.state_size();
    double x_diff = x_goal_(0) - human_traj.x(xs);
    double y_diff = x_goal_(1) - human_traj.x(xs+1);
    double d = std::sqrt(x_diff * x_diff + y_diff * y_diff) + reg_;
    double d3 = d * d * d;

    hess_x(0, 0) = -x_diff * x_diff / d3 + 1.0 / d;
    hess_x(0, 1) = -x_diff * y_diff / d3;
    hess_x(1, 0) = hess_x(0, 1);
    hess_x(1, 1) = -y_diff * y_diff / d3;

    hess = human_traj.Ju.middleRows(xs, 2).transpose() * hess_x * human_traj.Ju.middleRows(xs, 2);
}

//----------------------------------------------------------------------------------
double CollisionCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    // construct the pos diff vector
    Eigen::VectorXd x_diff(2 * robot_traj.horizon());

    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();

    for (int t = 0; t < robot_traj.horizon(); ++t) {
        x_diff(t*2) = robot_traj.x(t*nXr) - human_traj.x(t*nXh);
        x_diff(t*2+1) = robot_traj.x(t*nXr+1) - human_traj.x(t*nXh+1);
    }

    return GaussianCost::compute(x_diff, 2, robot_traj.horizon(), R_, R_);
}

//----------------------------------------------------------------------------------
void CollisionCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{

}

//----------------------------------------------------------------------------------
void CollisionCost::grad_ur(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{

}

//----------------------------------------------------------------------------------
void CollisionCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{

}

//----------------------------------------------------------------------------------
void CollisionCost::hessian_uh_ur(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{

}

}