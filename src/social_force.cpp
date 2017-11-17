//
// Created by yuhang on 11/16/17.
//

#include "hri_planner_sim/social_force.h"

namespace SocialForce {

vec2d social_force_interact(vec3d pose_human, vec3d vel_human,
                            vec3d pose_agent, vec3d vel_agent, std::vector<double> &params)
{
    vec2d force;

    // extract the parameters
    const double a = params[0];
    const double b = params[1];
    const double R = params[2];

    // the relative pose and velocity
    vec2d pos_rel = pose_human.head(2) - pose_agent.head(2);
    vec2d vel_rel = vel_human.head(2) - vel_agent.head(2);

    // calculate relative distance
    const double d = pos_rel.norm();

    // return 0 if dist is larger than threshold
    if (d > social_force_max_dist)
        return vec2d::Zero();

    return force;
}

vec2d social_force_hri(vec3d pose_human, vec3d vel_human,
                       vec3d pose_robot, vec2d robot, std::vector<double> &params)
{
    vec2d force;

    return force;
}

}