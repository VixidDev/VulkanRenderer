#include "Light.hpp"

Light::Light(LightType type, glm::vec3 pos, glm::vec3 direction, glm::vec3 colour, int intensity, float innerAngle, float outerAngle, bool shadowCasting) 
	: type(type), position(pos), direction(direction), colour(colour), intensity(intensity), innerAngle(innerAngle), outerAngle(outerAngle), shadowCasting(shadowCasting) {}

void Light::markDirty() {
	this->dirty = true;
}

void Light::markClean() {
	this->dirty = false;
}

LightType Light::getLightType() {
	return this->type;
}

glm::vec3 Light::getPosition() {
	return this->position;
}

glm::vec3 Light::getDirection() {
	return this->direction;
}

glm::vec3 Light::getColour() {
	return this->colour;
}

int Light::getIntensity() {
	return this->intensity;
}

float Light::getInnerAngle() {
	return this->innerAngle;
}

float Light::getOuterAngle() {
	return this->outerAngle;
}

bool Light::isShadowCasting() {
	return this->shadowCasting;
}

bool Light::isDirty() {
	return this->dirty;
}

void Light::toString() {}
