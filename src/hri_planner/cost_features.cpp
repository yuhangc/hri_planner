//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/8/2017
// Last revision: 3/25/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <iostream>
#include "hri_planner/cost_features.h"

namespace hri_planner {

// TODO: some of the second derivatives are not exact, maybe fix them later?

//----------------------------------------------------------------------------------
std::shared_ptr<FeatureHumanCost> FeatureHumanCost::create(const std::string &feature_type,
                                                           const std::vector<double> &args)
{
    if (feature_type == "Velocity") {
        return std::make_shared<HumanVelCost>();
    }
    else if (feature_type == "Acceleration") {
        return std::make_shared<HumanAccCost>();
    }
    else if (feature_type == "Goal") {
        Eigen::VectorXd x_goal(2);
        x_goal << args[0], args[1];
        return std::make_shared<HumanGoalCost>(x_goal);
    }
    else if (feature_type == "Collision") {
        return std::make_shared<CollisionCost>(args[0]);
    }
    else if (feature_type == "CollisionDynamic") {
        return std::make_shared<DynCollisionCost>(args[0], args[1], args[2]);
    }
    else {
        throw "Invalid feature type!";
    }
}

//----------------------------------------------------------------------------------
std::shared_ptr<FeatureRobotCost> FeatureRobotCost::create(const std::string &feature_type,
                                                           const std::vector<double> &args)
{
    if (feature_type == "Control") {
        return std::make_shared<RobotControlCost>();
    }
    else if (feature_type == "Goal") {
        Eigen::VectorXd x_goal(2);
        x_goal << args[0], args[1];
        return std::make_shared<RobotGoalCost>(x_goal);
    }
    else {
        throw "Invalid feature type!";
    }
}

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
    double x_diff = human_traj.x(xs) - x_goal_(0);
    double y_diff = human_traj.x(xs+1) - x_goal_(1);
    double d = std::sqrt(x_diff * x_diff + y_diff * y_diff) + reg_;

    grad_x(0) = x_diff / d;
    grad_x(1) = y_diff / d;

    grad = human_traj.Ju.middleRows(xs, 2).transpose() * grad_x;
}

//----------------------------------------------------------------------------------
void HumanGoalCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    Eigen::Matrix2d hess_x;
    int xs = human_traj.traj_state_size() - human_traj.state_size();
    double x_diff = human_traj.x(xs) - x_goal_(0);
    double y_diff = human_traj.x(xs+1) - x_goal_(1);
    double d = std::sqrt(x_diff * x_diff + y_diff * y_diff) + reg_;
    double d3 = d * d * d;

    hess_x(0, 0) = -x_diff * x_diff / d3 + 1.0 / d;
    hess_x(0, 1) = -x_diff * y_diff / d3;
    hess_x(1, 0) = hess_x(0, 1);
    hess_x(1, 1) = -y_diff * y_diff / d3 + 1.0 / d;

    hess = human_traj.Ju.middleRows(xs, 2).transpose() * hess_x * human_traj.Ju.middleRows(xs, 2);
}

//----------------------------------------------------------------------------------
double HumanObsCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    return 0;
}

//----------------------------------------------------------------------------------
void HumanObsCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{

}

//----------------------------------------------------------------------------------
void HumanObsCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{

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
    // construct the pos diff vector
    Eigen::VectorXd x_diff(2 * human_traj.horizon());

    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();

    for (int t = 0; t < robot_traj.horizon(); ++t) {
        x_diff(t*2) = human_traj.x(t*nXh) - robot_traj.x(t*nXr);
        x_diff(t*2+1) = human_traj.x(t*nXh+1) - robot_traj.x(t*nXr+1);
    }

    // compute gradient
    Eigen::VectorXd grad_x;
    GaussianCost::grad(x_diff, human_traj.state_size(), human_traj.horizon(), R_, R_, grad_x);

    grad = human_traj.Ju.transpose() * grad_x;
}

//----------------------------------------------------------------------------------
void CollisionCost::grad_ur(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    // construct the pos diff vector
    Eigen::VectorXd x_diff(2 * robot_traj.horizon());

    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();

    for (int t = 0; t < robot_traj.horizon(); ++t) {
        x_diff(t*2) = robot_traj.x(t*nXr) - human_traj.x(t*nXh);
        x_diff(t*2+1) = robot_traj.x(t*nXr+1) - human_traj.x(t*nXh+1);
    }

    // compute gradient
    Eigen::VectorXd grad_x;
    GaussianCost::grad(x_diff, robot_traj.state_size(), robot_traj.horizon(), R_, R_, grad_x);

    grad = robot_traj.Ju.transpose() * grad_x;
}

//----------------------------------------------------------------------------------
void CollisionCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    // construct the pos diff vector
    Eigen::VectorXd x_diff(2 * robot_traj.horizon());

    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();

    for (int t = 0; t < robot_traj.horizon(); ++t) {
        x_diff(t*2) = human_traj.x(t*nXh) - robot_traj.x(t*nXr);
        x_diff(t*2+1) = human_traj.x(t*nXh+1) - robot_traj.x(t*nXr+1);
    }

    // compute gradient
    Eigen::MatrixXd hess_x;
    GaussianCost::hessian(x_diff, human_traj.state_size(), human_traj.state_size(),
                          human_traj.horizon(), R_, R_, hess_x);

    hess = human_traj.Ju.transpose() * hess_x * human_traj.Ju;
}

//----------------------------------------------------------------------------------
void CollisionCost::hessian_uh_ur(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    // construct the pos diff vector
    Eigen::VectorXd x_diff(2 * robot_traj.horizon());

    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();

    for (int t = 0; t < robot_traj.horizon(); ++t) {
        x_diff(t*2) = human_traj.x(t*nXh) - robot_traj.x(t*nXr);
        x_diff(t*2+1) = human_traj.x(t*nXh+1) - robot_traj.x(t*nXr+1);
    }

    // compute gradient
    Eigen::MatrixXd hess_x;
    GaussianCost::hessian(x_diff, human_traj.state_size(), robot_traj.state_size(),
                          robot_traj.horizon(), R_, R_, hess_x);

    hess = human_traj.Ju.transpose() * (-hess_x) * robot_traj.Ju;
}

//----------------------------------------------------------------------------------
double DynCollisionCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    // compute the transformed coordinates
    int T = robot_traj.horizon();
    Eigen::VectorXd x_trans(2 * T);

    for (int t = 0; t < T; ++t) {
        int str = t * robot_traj.state_size();
        int sth = t * human_traj.state_size();
        double th = robot_traj.x(str + 2);

        // compute the center of gaussian cost
        Eigen::Vector2d xc;
        xc << robot_traj.x(str) + d_ * std::cos(th),
                robot_traj.x(str+1) + d_ * std::sin(th);

        // compute the transformed coordinate
        Eigen::Matrix2d rot;
        rot << std::cos(th), std::sin(th),
                -std::sin(th), std::cos(th);
        x_trans.segment(t*2, 2) = rot * (human_traj.x.segment(sth, 2) - xc);
    }

    return GaussianCost::compute(x_trans, 2, T, Rx_, Ry_);
}

//----------------------------------------------------------------------------------
void DynCollisionCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    // compute the transformed coordinates
    int T = robot_traj.horizon();
    Eigen::VectorXd x_trans(2 * T);

    // to cache the computation
    std::vector<Eigen::Matrix2d> rot;

    for (int t = 0; t < T; ++t) {
        int str = t * robot_traj.state_size();
        int sth = t * human_traj.state_size();
        double th = robot_traj.x(str + 2);

        // compute the center of gaussian cost
        Eigen::Vector2d xc;
        xc << robot_traj.x(str) + d_ * std::cos(th),
                robot_traj.x(str+1) + d_ * std::sin(th);

        // compute the transformed coordinate
        Eigen::Matrix2d rot_t;
        rot_t << std::cos(th), std::sin(th),
                -std::sin(th), std::cos(th);
        x_trans.segment(t*2, 2) = rot_t * (human_traj.x.segment(sth, 2) - xc);

        rot.push_back(rot_t);
    }

    // get gradient w.r.t. the transformed coordinate
    Eigen::VectorXd grad_x;
    GaussianCost::grad(x_trans, human_traj.state_size(), T, Rx_, Ry_, grad_x);

    // get gradient w.r.t. the original pose
    for (int t = 0; t < T; ++t) {
        int sth = t * human_traj.state_size();
        grad_x.segment(sth, 2) = rot[t].transpose() * grad_x.segment(sth, 2);
    }

    // compute gradient w.r.t. the control
    grad = human_traj.Ju.transpose() * grad_x;
}

//----------------------------------------------------------------------------------
void DynCollisionCost::grad_ur(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    // compute the transformed coordinates
    int T = robot_traj.horizon();
    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();
    Eigen::VectorXd x_trans(2 * T);

    // to cache the computation
    std::vector<Eigen::MatrixXd> Jxr;

    for (int t = 0; t < T; ++t) {
        int str = t * nXr;
        int sth = t * nXh;
        double th = robot_traj.x(str + 2);

        // compute the center of gaussian cost
        Eigen::Vector2d xc;
        xc << robot_traj.x(str) + d_ * std::cos(th),
                robot_traj.x(str+1) + d_ * std::sin(th);

        // compute the transformed coordinate
        Eigen::Matrix2d rot_t;
        rot_t << std::cos(th), std::sin(th),
                -std::sin(th), std::cos(th);
        x_trans.segment(t*2, 2) = rot_t * (human_traj.x.segment(sth, 2) - xc);

        // compute the jacobian w.r.t. xr
        Eigen::MatrixXd J(2, nXr);
//        J(0, 2) = x_trans(t*2+1);
//        J(1, 2) = -x_trans(t*2);
//
//        Eigen::MatrixXd Jp(2, nXr);
//        Jp << 1.0, 0.0, -d_ * std::sin(th),
//                0.0, 1.0, -d_ * std::cos(th);
//
//        J -= rot_t.transpose() * Jp;
        J.block(0, 0, 2, 2) = -rot_t;
        J(0, 2) = x_trans(t*2+1);
        J(1, 2) = -x_trans(t*2);
        Jxr.push_back(J);
    }

    // get gradient w.r.t. the transformed coordinate
    Eigen::VectorXd grad_x;
    GaussianCost::grad(x_trans, robot_traj.state_size(), T, Rx_, Ry_, grad_x);

    // get gradient w.r.t. the original pose
    for (int t = 0; t < T; ++t) {
        int str = t * nXr;
        grad_x.segment(str, nXr) = Jxr[t].transpose() * grad_x.segment(str, 2);
    }

    // compute gradient w.r.t. the control
    grad = robot_traj.Ju.transpose() * grad_x;
}

//----------------------------------------------------------------------------------
void DynCollisionCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    // compute the transformed coordinates
    int T = robot_traj.horizon();
    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();
    Eigen::VectorXd x_trans(2 * T);

    // to cache the computation
    std::vector<Eigen::Matrix2d> rot;

    for (int t = 0; t < T; ++t) {
        int str = t * nXr;
        int sth = t * nXh;
        double th = robot_traj.x(str + 2);

        // compute the center of gaussian cost
        Eigen::Vector2d xc;
        xc << robot_traj.x(str) + d_ * std::cos(th),
                robot_traj.x(str+1) + d_ * std::sin(th);

        // compute the transformed coordinate
        Eigen::Matrix2d rot_t;
        rot_t << std::cos(th), std::sin(th),
                -std::sin(th), std::cos(th);
        x_trans.segment(t*2, 2) = rot_t * (human_traj.x.segment(sth, 2) - xc);

        rot.push_back(rot_t);
    }

    // get hessian w.r.t. x_trans
    Eigen::MatrixXd hess_x;
    GaussianCost::hessian(x_trans, nXh, nXh, T, Rx_, Ry_, hess_x);

    // get gradient w.r.t. the original pose
    for (int t = 0; t < T; ++t) {
        int sth = t * nXh;
        hess_x.block(sth, sth, 2, 2) = rot[t].transpose() * hess_x.block(sth, sth, 2, 2) * rot[t];
    }

    // compute gradient w.r.t. the control
    hess = human_traj.Ju.transpose() * hess_x * human_traj.Ju;
}

//----------------------------------------------------------------------------------
void DynCollisionCost::hessian_uh_ur(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    // compute the transformed coordinates
    int T = robot_traj.horizon();
    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();
    Eigen::VectorXd x_trans(2 * T);

    // to cache the computation
    std::vector<Eigen::MatrixXd> Jxr;
    std::vector<Eigen::Matrix2d> Jxh;

    for (int t = 0; t < T; ++t) {
        int str = t * nXr;
        int sth = t * nXh;
        double th = robot_traj.x(str + 2);

        // compute the center of gaussian cost
        Eigen::Vector2d xc;
        xc << robot_traj.x(str) + d_ * std::cos(th),
                robot_traj.x(str+1) + d_ * std::sin(th);

        // compute the transformed coordinate
        Eigen::Matrix2d rot_t;
        rot_t << std::cos(th), std::sin(th),
                -std::sin(th), std::cos(th);
        x_trans.segment(t*2, 2) = rot_t * (human_traj.x.segment(sth, 2) - xc);

        Jxh.push_back(rot_t);

        // compute the jacobian w.r.t. xr
        Eigen::MatrixXd J(2, nXr);
        J.block(0, 0, 2, 2) = -rot_t;
        J(0, 2) = x_trans(t*2+1);
        J(1, 2) = -x_trans(t*2);

        Jxr.push_back(J);
    }

    // get gradient w.r.t. the transformed coordinate
    Eigen::VectorXd grad_x;
    Eigen::MatrixXd hess_x;
    GaussianCost::grad(x_trans, nXh, T, Rx_, Ry_, grad_x);
    GaussianCost::hessian(x_trans, nXh, nXr, T, Rx_, Ry_, hess_x);

    // get gradient w.r.t. the original pose
    for (int t = 0; t < T; ++t) {
        int str = t * nXr;
        int sth = t * nXh;
        grad_x.segment(sth, 2) = Jxh[t].transpose() * grad_x.segment(sth, 2);
        hess_x.block(sth, str, 2, nXr) = Jxh[t].transpose() * hess_x.block(sth, str, 2, 2) * Jxr[t];
        hess_x(sth, str+2) += grad_x(sth+1);
        hess_x(sth+1, str+2) -= grad_x(sth);
    }

    // compute gradient w.r.t. the control
    hess = human_traj.Ju.transpose() * hess_x * robot_traj.Ju;
}

//----------------------------------------------------------------------------------
double RobotControlCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    return robot_traj.u.squaredNorm();
}

//----------------------------------------------------------------------------------
void RobotControlCost::grad_ur(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    grad = 2.0 * robot_traj.u;
}

//----------------------------------------------------------------------------------
void RobotControlCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    grad.setZero();
}

//----------------------------------------------------------------------------------
double RobotGoalCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    int xs = robot_traj.traj_state_size() - robot_traj.state_size();
    double x_diff = robot_traj.x(xs) - x_goal_(0);
    double y_diff = robot_traj.x(xs+1) - x_goal_(1);

    return std::sqrt(x_diff * x_diff + y_diff * y_diff);
}

//----------------------------------------------------------------------------------
void RobotGoalCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    grad.setZero();
}

//----------------------------------------------------------------------------------
void RobotGoalCost::grad_ur(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    Eigen::Vector2d grad_x;
    int xs = robot_traj.traj_state_size() - robot_traj.state_size();
    double x_diff = robot_traj.x(xs) - x_goal_(0);
    double y_diff = robot_traj.x(xs+1) - x_goal_(1);
    double d = std::sqrt(x_diff * x_diff + y_diff * y_diff) + reg_;

    grad_x(0) = x_diff / d;
    grad_x(1) = y_diff / d;

    grad = robot_traj.Ju.middleRows(xs, 2).transpose() * grad_x;
}

}