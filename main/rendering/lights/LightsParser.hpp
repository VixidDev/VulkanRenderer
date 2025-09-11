#pragma once

#include <string>
#include <vector>

#include "Light.hpp"

namespace LightsParser {

	int parseLights(const std::string& filename, std::vector<Light>& lightsOut);

}