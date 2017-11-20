//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/16/2017
// Last revision: 11/19/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_SOCIAL_FORCE_H
#define HRI_PLANNER_SOCIAL_FORCE_H

#include <vector>
#include <cmath>

#include "Eigen/Dense"

//! namespace for all social force functions
namespace SocialForce {

const double social_force_max_dist = 5.0;

typedef Eigen::Vector2d vec2d;
typedef Eigen::Vector3d vec3d;

// helper function to calculate "cross product" for 2d vectors
double cross_prod_2d(const vec2d &vec1, const vec2d &vec2);

// calculate interactive social force
vec2d social_force_interact(vec3d &pose_human, vec3d &vel_human,
                            vec3d &pose_agent, vec3d &vel_agent, const std::vector<double> &params);

vec2d social_force_hri(vec3d &pose_human, vec3d &vel_human,
                       vec3d &pose_robot, vec2d &vel_robot, const std::vector<double> &params);

// force that drives people to goal
vec2d social_force_goal(vec3d &pose_human, vec3d &vel_human,
                        vec3d &pose_goal, const double k, const double vd);

}

#endif //HRI_PLANNER_SOCIAL_FORCE_H
