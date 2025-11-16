#pragma once

#include <string>
#include <vector>
#include <glm/vec3.hpp>

struct OBJModel {
	std::string name;
	std::vector<glm::vec3> vertices;
	std::vector<std::uint32_t> indices;
};