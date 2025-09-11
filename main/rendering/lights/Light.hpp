#pragma once

#include <glm/glm.hpp>

enum LightType {
	POINT,
	DIRECTIONAL,
	SPOT
};

class Light {
public:
	Light(LightType type, glm::vec3 pos, glm::vec3 direction, glm::vec3 colour = glm::vec3(1.0f));

	LightType getLightType();
	glm::vec3 getPosition();
	glm::vec3 getDirection();
	glm::vec3 getColour();

	void toString();

private:
	LightType type;
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 colour;
};