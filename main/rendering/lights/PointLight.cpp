#include "PointLight.hpp"

PointLight::PointLight(glm::vec3 pos, bool dynamic) : Light(pos, dynamic) {}

void PointLight::toString() {
	std::printf("Point light\n");
}