#pragma once

#include <glm/glm.hpp>
#include <string>

enum LightType {
	POINT,
	DIRECTIONAL,
	SPOT
};

class Light {
public:
	Light(
		LightType type, 
		glm::vec3 pos, 
		glm::vec3 direction,
		glm::vec3 colour, 
		int intensity, 
		float innerAngle, 
		float outerAngle, 
		bool shadowCasting
	);

	void markDirty() { this->dirty = true; }
	void markClean() { this->dirty = false; }

	LightType getLightType() const { return this->type; }
	glm::vec3 getPosition() const { return this->position; }
	glm::vec3 getDirection() const { return this->direction; }
	glm::vec3 getColour() const { return this->colour; }
	int getIntensity() const { return this->intensity; }
	float getInnerAngle() const { return this->innerAngle; }
	float getOuterAngle() const { return this->outerAngle; }
	bool isShadowCasting() const { return this->shadowCasting; }

	float getRadius() const;
	bool isEnabled() const { return this->enabled;  }
	bool isDirty() const { return this->dirty; }

	std::string toString() const;
private:
	LightType type;
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 colour;
	int intensity;
	float innerAngle;
	float outerAngle;
	bool shadowCasting;

	bool enabled = true;

	// Flag to mark that this light needs its shadow
	// map re-rendered
	bool dirty = true;
};