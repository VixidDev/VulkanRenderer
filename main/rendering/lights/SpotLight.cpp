#include "SpotLight.hpp"

SpotLight::SpotLight(glm::vec3 pos, glm::vec3 lookAt, bool dynamic) : Light(pos, dynamic), lookAt(lookAt) {}

void SpotLight::toString() {
	std::printf("Spot light\n");
}