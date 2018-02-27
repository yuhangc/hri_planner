//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/25/2017
// Last revision: 2/25/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

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

}

//----------------------------------------------------------------------------------
void SharedConfig::load_from_file(const std::string &file_path)
{

}

}