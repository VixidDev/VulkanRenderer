#pragma once

#include <string>
#include <vector>
#include <memory>

#include "Light.hpp"

using _Light = std::unique_ptr<Light>;

namespace LightsParser {

	int parseLights(const std::string& filename, std::vector<_Light>& lightsOut);

}