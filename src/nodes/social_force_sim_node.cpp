//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/20/2017
// Last revision: 11/20/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include "hri_planner_sim/social_force_sim.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "social_force_simulation");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    SocialForce::SocialForceSimGazebo sf_simulator(nh, pnh);

    const double sim_rate = 100;
    ros::Rate rate(sim_rate);
    while (!ros::isShuttingDown()) {
        ros::spinOnce();
        sf_simulator.update(1.0 / sim_rate);
        rate.sleep();
    }
}