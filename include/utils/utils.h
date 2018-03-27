//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/26/2017
// Last revision: 3/26/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_UTILS_H
#define HRI_PLANNER_UTILS_H

#include <vector>

#include <Eigen/Dense>

namespace utils {

#define PI 3.14159265359
#define PI2 6.28318530718

// common helper functions
double wrap_to_pi(double ang);

template<typename T>
T clamp(T x, T xmin, T xmax){
    if (x < xmin)
        return xmin;
    if (x > xmax)
        return xmax;
    return x;
};

// conversions
void EigenToVector(const Eigen::VectorXd& eigen_vec, std::vector<double>& std_vec);
void EigenToVector3(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2,
                    const Eigen::VectorXd& vec3, std::vector<double>& std_vec);
void VectorToEigen(const std::vector<double>& std_vec, Eigen::VectorXd& eigen_vec);

} // namespace

#endif //HRI_PLANNER_UTILS_H
