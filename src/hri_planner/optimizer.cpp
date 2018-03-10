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

#include "hri_planner/optimizer.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
TrajectoryOptimizer::TrajectoryOptimizer(unsigned int dim, const nlopt::algorithm& alg)
{
    optimizer_ = nlopt::opt(alg, dim);
}

//----------------------------------------------------------------------------------
void TrajectoryOptimizer::set_cost_function(SingleTrajectoryCost *cost)
{
    cost_ = std::shared_ptr<SingleTrajectoryCost>(cost);
}

//----------------------------------------------------------------------------------
void TrajectoryOptimizer::set_cost_function(std::shared_ptr<SingleTrajectoryCost> cost)
{
    cost_ = cost;
}

//----------------------------------------------------------------------------------
bool TrajectoryOptimizer::optimize(const Trajectory &traj_init, Trajectory &traj_opt)
{
    return true;
}

}