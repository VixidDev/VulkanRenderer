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

	void markDirty();
	void markClean();

	LightType getLightType();
	glm::vec3 getPosition();
	glm::vec3 getDirection();
	glm::vec3 getColour();
	int getIntensity();
	bool isDirty();

	void toString();
private:
	LightType type;
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 colour;
	int intensity;

	// Flag to mark that this light needs its shadow
	// map re-rendered
	bool dirty;
};