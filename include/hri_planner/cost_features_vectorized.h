//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/18/2018
// Last revision: 4/26/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COST_FEATURES_VECTORIZED_H
#define HRI_PLANNER_COST_FEATURES_VECTORIZED_H

#include <memory>

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

    static std::shared_ptr<FeatureVectorizedBase> create(const std::string &feature_type,
                                                         const std::vector<double> &args);

    // set additional data
    virtual void set_data(const void* data) = 0;
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

    // set additional data
    void set_data(const void* data) override {};

private:
    double R_;
};

//! gaussian collision avoidance feature centered in front of the robot
class DynCollisionCostVec: public FeatureVectorizedBase {
public:
    explicit DynCollisionCostVec(double Rx, double Ry, double d): Rx_(Rx), Ry_(Ry), d_(d) {};

    void compute(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::VectorXd& costs) override;
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Juh) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Jur) override;

    // set additional data
    void set_data(const void* data) override {};

private:
    double Rx_;
    double Ry_;
    double d_;
};

//! human effort feature
class HumanAccCostVec: public FeatureVectorizedBase {
public:
    void compute(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::VectorXd& costs) override;
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Juh) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Jur) override;

    // set additional data
    void set_data(const void* data) override {};
};

//! human goal feature
class HumanGoalCostVec: public FeatureVectorizedBase {
public:
    explicit HumanGoalCostVec(const Eigen::VectorXd& x_goal, double reg=1e-2): x_goal_(x_goal), reg_(reg) {};

    void compute(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::VectorXd& costs) override;
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Juh) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, Eigen::MatrixXd& Jur) override;

    // set additional data
    void set_data(const void* data) override {
        x_goal_ = *static_cast<const Eigen::VectorXd*>(data);
    };

private:
    Eigen::VectorXd x_goal_;
    double reg_;
};

}

#endif //HRI_PLANNER_COST_FEATURES_VECTORIZED_H
