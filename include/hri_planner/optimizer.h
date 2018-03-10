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

class TrajectoryOptimizer {
public:
    // constructor
    TrajectoryOptimizer(unsigned int dim, const nlopt::algorithm& alg=nlopt::LD_MMA);

    // set cost function
    void set_cost_function(SingleTrajectoryCost* cost);
    void set_cost_function(std::shared_ptr<SingleTrajectoryCost> cost);

    // optimize!
    bool optimize(const Trajectory& traj_init, Trajectory& traj_opt);

private:
    // non-linear optimizer
    nlopt::opt optimizer_;

    // pointer to cost function
    std::shared_ptr<SingleTrajectoryCost> cost_;

    // wrapper function for using nlopt interface
    float cost_func(const std::vector<float>& x, std::vector<double>& grad, void *cost_func_data);
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

    // optimize!
    bool optimize(const Trajectory& robot_traj_init, const Trajectory& human_traj_init,
                  Trajectory& robot_traj_opt, Trajectory& human_traj_opt);

private:
    // non-linear optimizer
    nlopt::opt optimizer_;

    // pointer to cost function
    std::shared_ptr<LinearCost> robot_cost_;
    std::shared_ptr<HumanCost> human_cost_;

    // wrapper cost function
    float cost_func(const std::vector<float>& x, std::vector<double>& grad, void *cost_func_data);

    // wrapper constraint function
    float constraint(const std::vector<float>& x, std::vector<double>& grad, void *constraint_data);
};

} // namespace

#endif //HRI_PLANNER_OPTIMIZER_H
