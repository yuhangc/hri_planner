//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2018
// Last revision: 4/18/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_PLANNER_H
#define HRI_PLANNER_PLANNER_H

#include <vector>
#include <unordered_map>
#include <memory>

#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>

#include "hri_planner/shared_config.h"
#include "hri_planner/trajectory.h"
#include "hri_planner/cost_features.h"
#include "hri_planner/cost_features_vectorized.h"
#include "hri_planner/cost_probabilistic.h"
#include "hri_planner/human_belief_model.h"
#include "hri_planner/optimizer.h"
#include "utils/utils.h"

#include "hri_planner/PlannedTrajectories.h"

namespace hri_planner {

class PlannerBase {
public:
    PlannerBase(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    virtual ~PlannerBase() = default;

    // main update function
    virtual void compute_plan(double t_max=-1) = 0;

    // publish the plan
    virtual void publish_plan() = 0;

    // reset the planner with new goals
    virtual void reset_planner(const Eigen::VectorXd& xr_goal, const Eigen::VectorXd& xh_goal,
                               const int intent, const std::string& ns="~") {
        xr_goal_ = xr_goal;
        xh_goal_ = xh_goal;
        intent_ = intent;
    }

    // simply clear any histories and reset flags
    virtual void reset_planner() {};

    // methods to send robot & human data in
    void set_robot_state(const Eigen::VectorXd& xr_meas, const Eigen::VectorXd& ur_meas) {
        xr_meas_ = xr_meas;
        ur_meas_ = ur_meas;
    }

    void set_human_state(const Eigen::VectorXd& xh_meas) {
        xh_meas_ = xh_meas;
    }

    // compute the closed-loop control for one time step
    void compute_steer_posq(const Eigen::VectorXd& xr, const Eigen::VectorXd& x_goal, Eigen::VectorXd& ur);

protected:
    // dimensions
    int T_;
    int nXh_;
    int nUh_;
    int nXr_;
    int nUr_;
    double dt_;

    // parameters for initializing trajectory
    double k_rho_;
    double k_v_;
    double k_alp_;
    double k_phi_;
    double gamma_;

    // control bounds
    std::vector<double> lb_ur_vec_;
    std::vector<double> ub_ur_vec_;

    // subscribers & publishers
    ros::NodeHandle nh_;

    // measurements
    Eigen::VectorXd xr_meas_;
    Eigen::VectorXd ur_meas_;
    Eigen::VectorXd xh_meas_;

    // goals for robot and human
    Eigen::VectorXd xr_goal_;
    Eigen::VectorXd xh_goal_;

    // true intent of the robot
    int intent_;

    // helper functions that can be useful to all derived classes
    virtual void update_init_guesses() = 0;

    void shift_control(const Eigen::VectorXd& u_in, Eigen::VectorXd& u_out, int dim, bool pad_zero);

    void generate_steer_posq(const Eigen::VectorXd& x0, const Eigen::VectorXd& x_goal, Eigen::VectorXd& ur);
    void generate_steer_acc(const Eigen::VectorXd& x0, const Eigen::VectorXd& x_goal, Eigen::VectorXd& uh);
};


class Planner: public PlannerBase {
public:
    // constructor
    Planner(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    // main update function
    void compute_plan(double t_max=-1) override;

    void publish_plan() override;

    // reset the planner with new goals
    void reset_planner(const Eigen::VectorXd& xr_goal, const Eigen::VectorXd& xh_goal,
                       const int intent, const std::string& ns="~") override;

    // simple reset
    void reset_planner() override;

    // get human prediction
    void get_human_pred(const int t, const int intent, Eigen::VectorXd& human_state);

private:
    // components
    std::shared_ptr<BeliefModelBase> belief_model_;
    std::shared_ptr<NestedOptimizerBase> optimizer_comm_;
    std::shared_ptr<NestedOptimizerBase> optimizer_no_comm_;

    // map to retrieve features by name
    std::unordered_map<std::string, std::shared_ptr<FeatureBase> > features_human_;
    std::unordered_map<std::string, std::shared_ptr<FeatureBase> > features_robot_;
    std::unordered_map<std::string, std::shared_ptr<FeatureVectorizedBase> > features_robot_int_;

    // recent explicit communicative action
    int acomm_;
    double tcomm_;

    // cost of communication
    double comm_cost_;

    // robot state and control
    Eigen::VectorXd xr_;
    Eigen::VectorXd ur_;

    // human state
    Eigen::VectorXd xh_;

    // the optimal plan
    Trajectory robot_traj_opt_;
    Trajectory human_traj_hp_opt_;
    Trajectory human_traj_rp_opt_;

    // initial guesses
    Trajectory robot_traj_init_;
    Trajectory human_traj_hp_init_;
    Trajectory human_traj_rp_init_;

    // control bounds
    std::vector<double> lb_uh_vec_;
    std::vector<double> ub_uh_vec_;

    // costs
    double cost_no_comm_;
    double cost_comm_;
    double cost_hp_no_comm_;
    double cost_rp_no_comm_;
    double cost_hp_comm_;
    double cost_rp_comm_;

    // whether to publish the full plan and belief/costs
    bool flag_publish_full_plan_;
    bool flag_publish_belief_cost_;

    // whether to generate initial guess from scratch
    bool flag_gen_init_guesses_;

    // subscribers & publishers
    ros::Publisher robot_ctrl_pub_;
    ros::Publisher comm_pub_;
    ros::Publisher plan_pub_;
    ros::Publisher belief_cost_pub_;

    // creation routines
    void create_belief_model(std::shared_ptr<BeliefModelBase>& belief_model);
//    void create_human_costs(std::shared_ptr<HumanCost>& human_cost_hp, std::shared_ptr<HumanCost>& human_cost_rp,
//                            std::shared_ptr<SingleTrajectoryCostHuman>& single_cost_hp,
//                            std::shared_ptr<SingleTrajectoryCostHuman>& single_cost_rp);
    void create_human_costs(std::vector<std::shared_ptr<SingleTrajectoryCostHuman> >& single_cost_hp,
                            std::vector<std::shared_ptr<SingleTrajectoryCostHuman> >& single_cost_rp,
                            int n);
    void create_robot_costs(std::vector<std::shared_ptr<ProbabilisticCostBase> >& robot_costs,
                            int n, const std::string& ns="~");
    void create_optimizer();

    // other helper functions
    void generate_init_guesses(Trajectory& robot_traj, Trajectory& human_traj_hp, Trajectory& human_traj_rp);
    void update_init_guesses() override;
};


class PlannerSimple: public PlannerBase {
public:
    PlannerSimple(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    // main update function
    void compute_plan(double t_max=-1) override;

    // publish plans
    void publish_plan() override;

    // reset the planner with new goals
    void reset_planner(const Eigen::VectorXd& xr_goal, const Eigen::VectorXd& xh_goal,
                       const int intent, const std::string& ns="~") override;

    // simple reset
    void reset_planner() override;

private:
    // the optimizer
    std::shared_ptr<TrajectoryOptimizer> optimizer_;

    // cost features
    std::unordered_map<std::string, std::shared_ptr<FeatureBase> > features_robot_;

    // robot state and control
    Eigen::VectorXd xr_;
    Eigen::VectorXd ur_;

    // optimal cost
    double cost_opt_;

    // the optimal plan
    Trajectory robot_traj_opt_;

    // initial guesses
    Trajectory robot_traj_init_;

    // whether to generate initial guess from scratch
    bool flag_gen_init_guesses_;

    bool flag_publish_full_plan_;

    // subscribers & publishers
    ros::Publisher robot_ctrl_pub_;
    ros::Publisher plan_pub_;

    // creation routines
    void create_robot_costs(std::shared_ptr<SingleTrajectoryCost>& robot_cost, const std::string& ns="~");
    void create_optimizer();

    // other helper functions
    void generate_init_guesses(Trajectory& robot_traj);
    void update_init_guesses() override;
};

} // namespace

#endif //HRI_PLANNER_PLANNER_H
