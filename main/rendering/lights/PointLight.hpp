#pragma once

#include "Light.hpp"

// A spherical point light that emits light in all directions
class PointLight : public Light {
public:
	PointLight(glm::vec3 pos, bool dynamic);

	void toString();
private:

};