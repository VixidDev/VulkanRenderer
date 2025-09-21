#pragma once

#include <glm/glm.hpp>

enum LightType {
	POINT,
	DIRECTIONAL,
	SPOT
};

class Light {
public:
	Light(LightType type, glm::vec3 pos, glm::vec3 direction, glm::vec3 colour, int intensity);

	LightType getLightType();
	glm::vec3 getPosition();
	glm::vec3 getDirection();
	glm::vec3 getColour();
	int getIntensity();

	void toString();

private:
	LightType type;
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 colour;
	int intensity;
};