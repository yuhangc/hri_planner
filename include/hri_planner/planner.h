//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2017
// Last revision: 3/28/2017
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

class Planner {
public:
    // constructor
    Planner(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    // main update function
    void compute_plan();

    // reset the planner with new goals
    void reset_planner(const Eigen::VectorXd& xr_goal, const Eigen::VectorXd& xh_goal, const int intent);

private:
    // dimensions
    int T_;
    int nXh_;
    int nUh_;
    int nXr_;
    int nUr_;
    double dt_;

    // components
    std::shared_ptr<BeliefModelBase> belief_model_;
    std::shared_ptr<NestedOptimizerBase> optimizer_;

    // map to retrieve features by name
    std::unordered_map<std::string, std::shared_ptr<FeatureBase> > features_human_;
    std::unordered_map<std::string, std::shared_ptr<FeatureBase> > features_robot_;

    // recent explicit communicative action
    int acomm_;
    double tcomm_;

    // cost of communication
    double comm_cost_;

    // true intent of the robot
    int intent_;

    // robot state and control
    Eigen::VectorXd xr_;
    Eigen::VectorXd ur_;

    // human state
    Eigen::VectorXd xh_;

    // measurements
    Eigen::VectorXd xr_meas_;
    Eigen::VectorXd ur_meas_;
    Eigen::VectorXd xh_meas_;

    // goals for robot and human
    Eigen::VectorXd xr_goal_;
    Eigen::VectorXd xh_goal_;

    // the optimal plan
    Trajectory robot_traj_opt_;
    Trajectory human_traj_hp_opt_;
    Trajectory human_traj_rp_opt_;

    // initial guesses
    Trajectory robot_traj_init_;
    Trajectory human_traj_hp_init_;
    Trajectory human_traj_rp_init_;

    // control bounds
    std::vector<double> lb_ur_vec_;
    std::vector<double> ub_ur_vec_;
    std::vector<double> lb_uh_vec_;
    std::vector<double> ub_uh_vec_;

    // whether to publish the full plan and belief/costs
    bool flag_publish_full_plan_;
    bool flag_publish_belief_cost_;

    // whether to generate initial guess from scratch
    bool flag_gen_init_guesses_;

    // subscribers & publishers
    ros::NodeHandle nh_;

    ros::Subscriber robot_state_sub_;
    ros::Subscriber robot_odom_sub_;
    ros::Subscriber human_state_sub_;

    ros::Publisher robot_ctrl_pub_;
    ros::Publisher comm_pub_;
    ros::Publisher plan_pub_;
    ros::Publisher belief_cost_pub_;

    // creation routines
    void create_belief_model(std::shared_ptr<BeliefModelBase>& belief_model);
    void create_human_costs(std::shared_ptr<HumanCost>& human_cost_hp, std::shared_ptr<HumanCost>& human_cost_rp,
                            std::shared_ptr<SingleTrajectoryCostHuman>& single_cost_hp,
                            std::shared_ptr<SingleTrajectoryCostHuman>& single_cost_rp);
    void create_robot_costs(std::shared_ptr<ProbabilisticCostBase>& robot_cost);
    void create_optimizer();

    // subscriber functions
    void robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg);
    void robot_odom_callback(const nav_msgs::OdometryConstPtr& odom_msg);
    void human_state_callback(const std_msgs::Float64MultiArrayConstPtr& state_msg);

    // other helper functions
    void generate_init_guesses(Trajectory& robot_traj, Trajectory& human_traj_hp, Trajectory& human_traj_rp);
    void update_init_guesses();

    void shift_control(const Eigen::VectorXd& u_in, Eigen::VectorXd& u_out, int dim, bool pad_zero);

    void generate_steer_posq(const Eigen::VectorXd& x0, const Eigen::VectorXd& x_goal, Eigen::VectorXd& ur);
    void generate_steer_acc(const Eigen::VectorXd& x0, const Eigen::VectorXd& x_goal, Eigen::VectorXd& uh);
};

} // namespace

#endif //HRI_PLANNER_PLANNER_H
