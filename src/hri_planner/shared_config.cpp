//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/25/2017
// Last revision: 2/27/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <fstream>

#include "hri_planner/shared_config.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
SharedConfig::SharedConfig(const std::string &config_file_path)
{
    if (config_file_path.empty())
        load();
    else
        load_from_file(config_file_path);
}

//----------------------------------------------------------------------------------
void SharedConfig::load()
{
    ros::param::param<int>("~planning_horizon", T, 10);
    ros::param::param<int>("~dimension_xh", nXh, 4);
    ros::param::param<int>("~dimension_uh", nUh, 2);
    ros::param::param<int>("~dimension_xr", nXr, 3);
    ros::param::param<int>("~dimension_ur", nUr, 2);
    ros::param::param<double>("~time_step", dt, 0.5);
}

//----------------------------------------------------------------------------------
void SharedConfig::load_from_file(const std::string &file_path)
{
    std::ifstream config_file(file_path, std::ifstream::binary);
    Json::Value root;

    config_file >> root;

    T = root["planning_horizon"].asInt();
    nXh = root["dimension_xh"].asInt();
    nUh = root["dimension_uh"].asInt();
    nXr = root["dimension_xr"].asInt();
    nUr = root["dimension_ur"].asInt();
}

}