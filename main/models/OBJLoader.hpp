#pragma once

#include "OBJModel.hpp"
#include "tiny_obj_loader.h"

namespace OBJLoader {

	int loadFromFile(const std::string& filename, OBJModel& modelOut);

}