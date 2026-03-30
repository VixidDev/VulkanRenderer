#pragma once

#include "Light.hpp"

#include <vector>

namespace Lights {

	int parse(const std::string& filename);

	std::vector<Light>& getPointLights();
	std::vector<Light>& getDirectionalLights();
	std::vector<Light>& getSpotLights();

	constexpr std::size_t getNbPointLights();
	constexpr std::size_t getNbDirectionalLights();
	constexpr std::size_t getNbSpotLights();

	constexpr std::size_t getNbShadowPointLights();
	constexpr std::size_t getNbShadowDirectionalLights();
	constexpr std::size_t getNbShadowSpotLights();

}