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

#include "utils/utils.h"

namespace utils {

//----------------------------------------------------------------------------------
double wrap_to_pi(double ang)
{
    while (ang >= PI)
        ang -= PI2;
    while (ang < -PI)
        ang += PI2;

    return ang;
}

//----------------------------------------------------------------------------------
void EigenToVector(const Eigen::VectorXd& eigen_vec, std::vector<double>& std_vec)
{
    std_vec.assign(eigen_vec.data(), eigen_vec.data() + eigen_vec.size());
}

//----------------------------------------------------------------------------------
void EigenToVector3(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2,
                    const Eigen::VectorXd& vec3, std::vector<double>& std_vec)
{
    std_vec.assign(vec1.data(), vec1.data() + vec1.size());
    std_vec.insert(std_vec.end(), vec2.data(), vec2.data() + vec2.size());
    std_vec.insert(std_vec.end(), vec3.data(), vec3.data() + vec3.size());
}

//----------------------------------------------------------------------------------
void VectorToEigen(const std::vector<double>& std_vec, Eigen::VectorXd& eigen_vec)
{
    eigen_vec = Eigen::Map<Eigen::VectorXd>(const_cast<double*>(std_vec.data()), std_vec.size());
}

}