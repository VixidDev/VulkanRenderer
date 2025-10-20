#pragma once

#include <glm/glm.hpp>

enum LightType {
	POINT,
	DIRECTIONAL,
	SPOT
};

class Light {
public:
	Light(LightType type, glm::vec3 pos, glm::vec3 direction, glm::vec3 colour, int intensity, float innerAngle = 0.0f, float outerAngle = 0.0f);

	void markDirty();
	void markClean();

	LightType getLightType();
	glm::vec3 getPosition();
	glm::vec3 getDirection();
	glm::vec3 getColour();
	int getIntensity();
	float getInnerAngle();
	float getOuterAngle();
	bool isDirty();

	void toString();
private:
	LightType type;
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 colour;
	int intensity;
	float innerAngle;
	float outerAngle;

	// Flag to mark that this light needs its shadow
	// map re-rendered
	bool dirty;
};