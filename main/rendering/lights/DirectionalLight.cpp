#include "DirectionalLight.hpp"

DirectionalLight::DirectionalLight(glm::vec3 pos, glm::vec3 lookAt, bool dynamic) : Light(pos, dynamic), lookAt(lookAt) {}

void DirectionalLight::toString() {
	std::printf("Directional light\n");
}
