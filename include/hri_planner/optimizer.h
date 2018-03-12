//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/9/2017
// Last revision: 3/9/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_OPTIMIZER_H
#define HRI_PLANNER_OPTIMIZER_H

#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <nlopt.hpp>

#include "hri_planner/trajectory.h"
#include "hri_planner/costs.h"

namespace hri_planner {

// helper functions
void EigenToVector(const Eigen::VectorXd& eigen_vec, std::vector<double>& std_vec);
void VectorToEigen(const std::vector<double>& std_vec, Eigen::VectorXd& eigen_vec);

class TrajectoryOptimizer {
public:
    // constructor
    TrajectoryOptimizer(unsigned int dim, const nlopt::algorithm& alg=nlopt::LD_MMA);

    // set cost function
    void set_cost_function(SingleTrajectoryCost* cost);
    void set_cost_function(std::shared_ptr<SingleTrajectoryCost> cost);

    void set_bounds(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

    // optimize!
    bool optimize(const Eigen::VectorXd& x0, const Trajectory& traj_init, Trajectory& traj_opt);

private:
    // non-linear optimizer
    nlopt::opt optimizer_;

    // pointer to cost function
    std::shared_ptr<SingleTrajectoryCost> cost_;

    // an trajectory object to facilitate cost computation
    std::unique_ptr<Trajectory> traj_;

    // initial state
    Eigen::VectorXd x0_;

    // lower and upper bounds
    Eigen::VectorXd lb_;
    Eigen::VectorXd ub_;

    // wrapper function for using nlopt interface
    double cost_func(const std::vector<double>& u, std::vector<double>& grad);
    static double cost_wrapper(const std::vector<double>& u, std::vector<double>& grad, void* data);
};

class NestedTrajectoryOptimizer {
public:
    // constructor
    NestedTrajectoryOptimizer(unsigned int dim, const nlopt::algorithm& alg=nlopt::LD_MMA);

    // set cost functions
    void set_robot_cost(LinearCost* cost);
    void set_robot_cost(std::shared_ptr<LinearCost> cost);

    void set_human_cost(HumanCost* cost);
    void set_human_cost(std::shared_ptr<HumanCost> cost);

    void set_bounds(const Eigen::VectorXd& lb_ur, const Eigen::VectorXd& ub_ur,
                    const Eigen::VectorXd& lb_uh, const Eigen::VectorXd& ub_uh);

    // optimize!
    bool optimize(const Eigen::VectorXd& xr0, const Eigen::VectorXd& xh0,
                  const Trajectory& robot_traj_init, const Trajectory& human_traj_init,
                  Trajectory& robot_traj_opt, Trajectory& human_traj_opt);

private:
    // non-linear optimizer
    nlopt::opt optimizer_;

    // pointer to cost function
    std::shared_ptr<LinearCost> robot_cost_;
    std::shared_ptr<HumanCost> human_cost_;

    // an trajectory object to facilitate cost computation
    std::unique_ptr<Trajectory> robot_traj_;
    std::unique_ptr<Trajectory> human_traj_;

    // initial state
    Eigen::VectorXd xr0_;
    Eigen::VectorXd xh0_;

    // lower and upper bounds
    Eigen::VectorXd lb_ur_;
    Eigen::VectorXd ub_ur_;
    Eigen::VectorXd lb_uh_;
    Eigen::VectorXd ub_uh_;

    // wrapper cost function
    double cost_func(const std::vector<double>& u, std::vector<double>& grad);
    double cost_wrapper(const std::vector<double>& u, std::vector<double>& grad, void *cost_func_data);

    // wrapper constraint function
    double constraint(const std::vector<double>& u, std::vector<double>& grad);
    double constraint_wrapper(const std::vector<double>& u, std::vector<double>& grad, void *constraint_data);
};

} // namespace

#endif //HRI_PLANNER_OPTIMIZER_H
