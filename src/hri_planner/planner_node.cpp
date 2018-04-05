//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2018
// Last revision: 3/31/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include "hri_planner/planner_node.h"

//----------------------------------------------------------------------------------
PlannerNode::PlannerNode(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    // get parameters
    ros::param::param<double>("~planner/planner_rate", planning_rate_, 2.0);
    ros::param::param<double>("~planner/state_machine_rate", state_machine_rate_, 1000);
    ros::param::param<std::string>("~planner/planner_mode", mode_, "simulation");

    int nXh, nUh, nXr, nUr;
    ros::param::param<int>("~dimension/nXh", nXh, 4);
    ros::param::param<int>("~dimension/nUh", nUh, 2);
    ros::param::param<int>("~dimension/nXr", nXr, 3);
    ros::param::param<int>("~dimension/nUr", nUr, 2);

    // create planner
    planner_interactive_ = std::make_shared<hri_planner::Planner>(nh, pnh);

    // measurements
    xr_meas_.setZero(nXr);
    ur_meas_.setZero(nUr);
    xh_meas_.setZero(nXh);

    // create subscribers
    goal_sub_ = nh.subscribe<std_msgs::Float64MultiArray>("/planner/set_goal", 1,
                                                          &PlannerNode::goal_callback, this);
    planner_ctrl_sub_ = nh.subscribe<std_msgs::Bool>("/planner/pause", 1,
                                                     &PlannerNode::pause_planner_callback, this);

    robot_state_sub_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 1,
                                                            &PlannerNode::robot_state_callback, this);

    robot_odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &PlannerNode::robot_odom_callback, this);

    human_state_sub_ = nh_.subscribe<std_msgs::Float64MultiArray>("/tracked_human", 1,
                                                                  &PlannerNode::human_state_callback, this);
}

//----------------------------------------------------------------------------------
void PlannerNode::run()
{
    // loop and a simple state machine
    PlannerStates planner_state = Idle;
    flag_start_planning_ = false;
    flag_pause_planning_ = true;

    // two rates
    ros::Rate rate_fast(state_machine_rate_);
    ros::Rate rate_slow(planning_rate_);

    while (!ros::isShuttingDown()) {
        switch (planner_state) {
            case Idle:
                ROS_INFO("In state Idle");

                rate_fast.reset();
                while (!flag_start_planning_ && !ros::isShuttingDown()) {
                    ros::spinOnce();
                    rate_fast.sleep();
                }

                flag_start_planning_ = false;
                planner_interactive_->reset_planner(xr_goal_, xh_goal_, intent_);

                planner_state = Planning;

                break;
            case Pausing:
                ROS_INFO("In state Pausing");

                rate_fast.sleep();
                while (flag_pause_planning_ && !ros::isShuttingDown()) {
                    ros::spinOnce();
                    rate_fast.sleep();
                }

                flag_pause_planning_ = true;
                planner_state = Planning;

                break;
            case Planning:
                ROS_INFO("In state Planning");

                rate_slow.reset();
                while (!ros::isShuttingDown()) {
                    ros::spinOnce();

                    // update planner measurements
                    planner_interactive_->set_robot_state(xr_meas_, ur_meas_);
                    planner_interactive_->set_human_state(xh_meas_);

                    // compute plan
                    auto t_s = ros::Time::now();
                    planner_interactive_->compute_plan();
                    ros::Duration t_plan = ros::Time::now() - t_s;
                    std::cout << "time spent for planning is: " << t_plan.toSec() << "s" << std::endl;

                    // publish plan
                    planner_interactive_->publish_plan();

                    rate_slow.sleep();

                    if (mode_ == "simulation") {
                        planner_state = Pausing;
                        break;
                    }
                    else {
                        // check for exit flag or goal reached
                        ROS_ERROR("To be implemented!");
                    }
                }

                break;
            case PlanningNoHuman:
                ROS_ERROR("To be implemented!");
                break;
        }
    }
}

//----------------------------------------------------------------------------------
void PlannerNode::goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg)
{
    int dim;
    ros::param::param<int>("~dimension/dim_goal", dim, 2);

    ROS_INFO("Received new goal, reset planner...");

    xr_goal_.resize(dim);
    xh_goal_.resize(dim);

    for (int i = 0; i < dim; ++i) {
        xr_goal_(i) = goal_msg->data[i];
        xh_goal_(i) = goal_msg->data[i+dim];
    }

    // the last value of goal is intent
    intent_ = static_cast<int>(goal_msg->data[dim << 1]);

    flag_start_planning_ = true;
}

//----------------------------------------------------------------------------------
void PlannerNode::pause_planner_callback(const std_msgs::BoolConstPtr& msg)
{
    flag_pause_planning_ = msg->data;
}

//----------------------------------------------------------------------------------
void PlannerNode::robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &pose_msg)
{
    xr_meas_(0) = pose_msg->pose.pose.position.x;
    xr_meas_(1) = pose_msg->pose.pose.position.y;

    // find rotation
    auto &q = pose_msg->pose.pose.orientation;
    double siny = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);

    xr_meas_(2) = std::atan2(siny, cosy);
}

//----------------------------------------------------------------------------------
void PlannerNode::robot_odom_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    ur_meas_(0) = odom_msg->twist.twist.linear.x;
    ur_meas_(1) = odom_msg->twist.twist.angular.z;
}

//----------------------------------------------------------------------------------
void PlannerNode::human_state_callback(const std_msgs::Float64MultiArrayConstPtr& state_msg)
{
    xh_meas_(0) = state_msg->data[0];
    xh_meas_(1) = state_msg->data[1];
    xh_meas_(2) = state_msg->data[2];
    xh_meas_(3) = state_msg->data[3];
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "hri_planner");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // create the planner and run
    PlannerNode planner_node(nh, pnh);
    planner_node.run();

    return 0;
}