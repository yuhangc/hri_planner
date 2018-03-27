//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2017
// Last revision: 3/26/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <string>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>

#include "hri_planner/planner.h"

// enum for state machine
enum PlannerStates {
    Idle,
    Planning,
    PlanningNoHuman,
    Waiting
};

bool flag_start_planning;
bool flag_pause_planning;
Eigen::VectorXd xr_goal;
Eigen::VectorXd xh_goal;

void goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg)
{
    int dim;
    ros::param::param<int>("~dimension/dim_goal", dim, 2);

    xr_goal.resize(dim);
    xh_goal.resize(dim);

    for (int i = 0; i < dim; ++i) {
        xr_goal(i) = goal_msg->data[i];
        xh_goal(i) = goal_msg->data[i+dim];
    }

    flag_start_planning = true;
}

void human_state_callback(const std_msgs::Float64MultiArrayConstPtr& state_msg)
{
    flag_pause_planning = false;
}

int main(int argc, char** argv)
{

    ros::init(argc, argv, "hri_planner");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // the planner object
    hri_planner::Planner planner(nh, pnh);

    // subscribers
    auto goal_sub = nh.subscribe<std_msgs::Float64MultiArray>("/planner/set_goal", 1, &goal_callback);
    auto human_state_sub = nh.subscribe<std_msgs::Float64MultiArray>("/tracked_human", 1, &human_state_callback);

    // get rate
    double planner_rate;
    ros::param::param<double>("~planner_rate", planner_rate, 2.0);

    // operating mode
    std::string mode;
    ros::param::param<std::string>("~planner_mode", mode, "simulation");

    ros::Rate rate(planner_rate);

    // loop and a simple state machine
    PlannerStates planner_state = Idle;
    flag_start_planning = false;
    flag_pause_planning = true;

    while (!ros::isShuttingDown()) {
        switch (planner_state) {
            case Idle:
                if (flag_start_planning) {
                    flag_start_planning = false;
                    planner.reset_planner(xr_goal, xh_goal);

                    planner_state = Planning;
                }
            case Waiting:
                if (!flag_pause_planning) {
                    flag_pause_planning = true;
                    planner_state = Planning;
                }
            case Planning:
                planner.compute_plan();

                if (mode == "simulation") {
                    planner_state = Waiting;
                }
                else {
                    // check for exit flag or goal reached
                    ROS_ERROR("To be implemented!");
                }
                break;
            case PlanningNoHuman:
                ROS_ERROR("To be implemented!");
                break;
        }

        rate.sleep();
    }

    return 0;
}