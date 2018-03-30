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

#include <string>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

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
int intent;

void goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg)
{
    int dim;
    ros::param::param<int>("~dimension/dim_goal", dim, 2);

    ROS_INFO("Received new goal, reset planner...");

    xr_goal.resize(dim);
    xh_goal.resize(dim);

    for (int i = 0; i < dim; ++i) {
        xr_goal(i) = goal_msg->data[i];
        xh_goal(i) = goal_msg->data[i+dim];
    }

    // the last value of goal is intent
    intent = static_cast<int>(goal_msg->data[dim << 1]);

    flag_start_planning = true;
}

void pause_planner_callback(const std_msgs::BoolConstPtr& msg)
{
    flag_pause_planning = msg->data;
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
    auto pause_planner_sub = nh.subscribe<std_msgs::Bool>("/planner/pause", 1, &pause_planner_callback);

    // get rate
    double planner_rate;
    double state_machine_rate;
    ros::param::param<double>("~planner/planner_rate", planner_rate, 2.0);
    ros::param::param<double>("~planner/state_machine_rate", state_machine_rate, 1000);

    // operating mode
    std::string mode;
    ros::param::param<std::string>("~planner/planner_mode", mode, "simulation");

    ros::Rate rate_slow(planner_rate);
    ros::Rate rate_fast(state_machine_rate);

    // loop and a simple state machine
    PlannerStates planner_state = Idle;
    flag_start_planning = false;
    flag_pause_planning = true;

    while (!ros::isShuttingDown()) {
        switch (planner_state) {
            case Idle:
                ROS_INFO("In state Idle");
                while (!flag_start_planning && !ros::isShuttingDown()) {
                    ros::spinOnce();
                    rate_fast.sleep();
                }

                flag_start_planning = false;
                planner.reset_planner(xr_goal, xh_goal, intent);

                planner_state = Planning;

                break;
            case Waiting:
                ROS_INFO("In state Waiting");
                while (flag_pause_planning && !ros::isShuttingDown()) {
                    ros::spinOnce();
                    rate_fast.sleep();
                }

                flag_pause_planning = true;
                planner_state = Planning;

                break;
            case Planning:
                ROS_INFO("In state Planning");
                while (!ros::isShuttingDown()) {
                    ros::spinOnce();
                    planner.compute_plan();
                    rate_slow.sleep();

                    if (mode == "simulation") {
                        planner_state = Waiting;
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

    return 0;
}