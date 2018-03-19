//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/18/2017
// Last revision: 3/18/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COST_FEATURES_VECTORIZED_H
#define HRI_PLANNER_COST_FEATURES_VECTORIZED_H

#include <Eigen/Dense>

#include "hri_planner/trajectory.h"

namespace hri_planner {

// cost feature that produce a sequence of costs rather than a sum
class FeatureVectorizedBase {
public:
    // virtual destructor
    virtual ~FeatureVectorizedBase() = default;

    virtual void compute(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::VectorXd& costs) = 0;
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Juh) = 0;
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Jur) = 0;
};

//! vectorized gaussian cost
class GaussianCostVec {
public:
    static void compute(const Eigen::VectorXd& x, const int nX, const int T,
                        const double a, const double b, Eigen::VectorXd& costs);
    static void grad(const Eigen::VectorXd& x, const int nX, const int T,
                     const double a, const double b, Eigen::VectorXd& grad);
};

//! gaussian collision avoidance feature
class CollisionCostVec: public FeatureVectorizedBase {
public:
    explicit CollisionCostVec(double R): R_(R) {};

    void compute(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::VectorXd& costs) override;
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Juh) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Jur) override;

private:
    double R_;
};

//! human effort feature
class HumanAccVec: public FeatureVectorizedBase {
    void compute(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::VectorXd& costs) override;
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Juh) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Jur) override;
};

}

#endif //HRI_PLANNER_COST_FEATURES_VECTORIZED_H
