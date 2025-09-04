#pragma once

#include "Light.hpp"

class DirectionalLight : public Light {
public:
	DirectionalLight(glm::vec3 pos, glm::vec3 lookAt, bool dynamic);

	void toString();
private:
	glm::vec3 lookAt;
};