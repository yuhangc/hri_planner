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

#include "social_force/social_force.h"

namespace SocialForce {

//-----------------------------------------------------------------------------------------
double cross_prod_2d(const vec2d &vec1, const vec2d &vec2)
{
    return vec1(0) * vec2(1) - vec1(1) * vec2(0);
}

//-----------------------------------------------------------------------------------------
void clip_vec(Eigen::Ref<Eigen::VectorXd> vec, const double max_norm)
{
    double vec_norm = vec.norm();

    if (vec_norm > max_norm)
        vec = vec * max_norm / vec_norm;
}

//-----------------------------------------------------------------------------------------
vec2d social_force_interact(const vec3d &pose_human, const vec3d &vel_human,
                            const vec3d &pose_agent, const vec3d &vel_agent,
                            const std::vector<double> &params)
{
    vec2d force;

    // extract the parameters
    const double a = params[0];
    const double b = params[1];
    const double R = params[2];

    // the relative pose and velocity
    vec3d pos_tmp = pose_human - pose_agent;
    vec3d vel_tmp = vel_human - vel_agent;
    vec2d pos_rel = pos_tmp.head(2);
    vec2d vel_rel = vel_tmp.head(2);

    // calculate relative distance
    const double d = pos_rel.norm();

    // return 0 if dist is larger than threshold
    if (d > social_force_max_dist)
        return vec2d::Zero();

    // check which direction is closest
    const double dir_rel = pos_rel.dot(vel_rel);
    if (dir_rel > 0) {
        // calculate cross product and minimum distance
        const double cross_prod = cross_prod_2d(pos_rel, vel_rel);
        const double vmag = vel_rel.norm();
        const double dmin = std::abs(cross_prod / vmag);

        // check if dmin is within threshold
        if (dmin < R) {
            // social force magnitude
            const double fmag = a * std::exp(-d / b);

            // social force direction
            if (cross_prod > 0) {
                force << -vel_rel(1), vel_rel(0);
            } else {
                force << vel_rel(1), -vel_rel(0);
            }

            force = fmag / vmag * force;
        } else
            return vec2d::Zero();
    } else {
        if (d < R) {
            // social force magnitude
            const double fmag = a * std::exp(-d / b);

            // social force direction
            force = pos_rel / d * fmag;
        } else
            return vec2d::Zero();
    }

    return force;
}

//-----------------------------------------------------------------------------------------
vec2d social_force_hri(const vec3d &pose_human, const vec3d &vel_human,
                       const vec3d &pose_robot, const vec2d &vel_robot,
                       const std::vector<double> &params)
{
    double v_hat = vel_robot(0) + params[3];
    if (v_hat < 0)
        v_hat = 0.0;

    const double th = pose_robot(2);
    vec3d vel_agent(v_hat * std::cos(th), v_hat * std::sin(th), vel_robot(1));

    return social_force_interact(pose_human, vel_human, pose_robot, vel_agent, params);
}

//-----------------------------------------------------------------------------------------
vec2d social_force_goal(const vec3d &pose_human, const vec3d &vel_human,
                        const vec3d &pose_goal, const double k, const double vd)
{
    vec3d pos_tmp = pose_goal - pose_human;
    const vec2d pos_rel = pos_tmp.head(2);

    vec2d vel;
    vel << vel_human(0), vel_human(1);

    const double d = pos_rel.norm();

    if (d < sf_goal_reached_th)
        return k * (-vel);
    else
        return k * (pos_rel / pos_rel.norm() * vd - vel);
}

}