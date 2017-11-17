//
// Created by yuhang on 11/16/17.
//

#ifndef HRI_PLANNER_SOCIAL_FORCE_H
#define HRI_PLANNER_SOCIAL_FORCE_H

#include <vector>

#include "Eigen/Dense"

//! namespace for all social force functions
namespace SocialForce {

const double social_force_max_dist = 5.0;

typedef Eigen::Vector2d vec2d;
typedef Eigen::Vector3d vec3d;

vec2d social_force_interact(vec3d pose_human, vec3d vel_human,
                            vec3d pose_agent, vec3d vel_agent, std::vector<double> &params);

vec2d social_force_hri(vec3d pose_human, vec3d vel_human,
                       vec3d pose_robot, vec2d robot, std::vector<double> &params);

}

#endif //HRI_PLANNER_SOCIAL_FORCE_H
