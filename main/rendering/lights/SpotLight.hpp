#pragma once

#include "Light.hpp"

class SpotLight : public Light {
public:
	SpotLight(glm::vec3 pos, glm::vec3 lookAt, bool dynamic);

	void toString();
private:
	glm::vec3 lookAt;
};