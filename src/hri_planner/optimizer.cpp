//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/9/2017
// Last revision: 3/13/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <utility>

#include "hri_planner/optimizer.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
void EigenToVector(const Eigen::VectorXd& eigen_vec, std::vector<double>& std_vec)
{
    std_vec.assign(eigen_vec.data(), eigen_vec.data() + eigen_vec.size());
}

//----------------------------------------------------------------------------------
void VectorToEigen(const std::vector<double>& std_vec, Eigen::VectorXd& eigen_vec)
{
    eigen_vec = Eigen::Map<Eigen::VectorXd>(const_cast<double*>(std_vec.data()), std_vec.size());
}

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
    cost_ = std::move(cost);
}

//----------------------------------------------------------------------------------
void TrajectoryOptimizer::set_bounds(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
{
    lb_ = lb;
    ub_ = ub;
}

//----------------------------------------------------------------------------------
bool TrajectoryOptimizer::optimize(const Trajectory &traj_init, Trajectory &traj_opt)
{
    // recreate the trajectory object with trajectory data
    traj_.reset(new Trajectory(traj_init.dyn_type, traj_init.horizon(), traj_init.dt()));
    traj_->x0 = traj_init.x0;

    // set lower and upper bounds
    std::vector<double> lb;
    std::vector<double> ub;

    for (int t = 0; t < traj_init.horizon(); ++t) {
        for (int i = 0; i < traj_init.control_size(); ++i) {
            lb.push_back(lb_(i));
            ub.push_back(ub_(i));
        }
    }

    optimizer_.set_lower_bounds(lb);
    optimizer_.set_upper_bounds(ub);

    // set cost function
    optimizer_.set_min_objective(cost_wrapper, this);

    // set tolerance
    optimizer_.set_xtol_abs(1e-2);

    // initial condition
    std::vector<double> u_opt;
    EigenToVector(traj_init.u, u_opt);

    // optimizer!
    double min_cost;
    optimizer_.optimize(u_opt, min_cost);

    // send result back
    traj_opt.x0 = traj_init.x0;
    VectorToEigen(u_opt, traj_opt.u);
    traj_opt.compute();

    return true;
}

//----------------------------------------------------------------------------------
double TrajectoryOptimizer::cost_func(const std::vector<double> &u, std::vector<double> &grad)
{
    // re-compute the trjectory
    Eigen::Map<const Eigen::VectorXd> u_new(u.data(), u.size());

    traj_->update(u_new);
    traj_->compute_jacobian();

    // compute the cost and gradient
    Eigen::VectorXd grad_vec(u_new.size());
    cost_->grad(*traj_, grad_vec);

    // covert eigen vec to std vector
    EigenToVector(grad_vec, grad);

    // return the cost
    return cost_->compute(*traj_);
}

//----------------------------------------------------------------------------------
double TrajectoryOptimizer::cost_wrapper(const std::vector<double> &u, std::vector<double> &grad, void *data)
{
    return reinterpret_cast<TrajectoryOptimizer *>(data)->cost_func(u, grad);
}

//----------------------------------------------------------------------------------
NestedTrajectoryOptimizer::NestedTrajectoryOptimizer(unsigned int dim, const nlopt::algorithm &alg)
{

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(HumanCost *cost)
{

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(std::shared_ptr<HumanCost> cost)
{

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_robot_cost(LinearCost *cost)
{

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_robot_cost(std::shared_ptr<LinearCost> cost)
{

}

//----------------------------------------------------------------------------------
bool NestedTrajectoryOptimizer::optimize(const Eigen::VectorXd &xr0, const Eigen::VectorXd &xh0,
                                         const Trajectory &robot_traj_init, const Trajectory &human_traj_init,
                                         Trajectory &robot_traj_opt, Trajectory &human_traj_opt)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::cost_func(const std::vector<double> &u, std::vector<double> &grad)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::cost_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                              void *cost_func_data)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::constraint(const std::vector<double> &u, std::vector<double> &grad)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::constraint_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                                    void *constraint_data)
{

}

} // namespace