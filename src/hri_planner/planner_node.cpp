//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2017
// Last revision: 3/24/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <ros/ros.h>

int main(int argc, char** argv)
{

    ros::init(argc, argv, "hri_planner");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    return 0;
}