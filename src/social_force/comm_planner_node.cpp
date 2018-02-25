//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/28/2017
// Last revision: 11/28/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include "social_force/comm_planner.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "communication_planner");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    HRIPlanner::CommPlanner comm_planner(nh, pnh);

    comm_planner.run();
}