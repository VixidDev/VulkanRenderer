#include "Light.hpp"

Light::Light(LightType type, glm::vec3 pos, glm::vec3 direction, glm::vec3 colour, int intensity) 
	: type(type), position(pos), direction(direction), colour(colour), intensity(intensity) {}

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

bool Light::isDirty() {
	return this->dirty;
}

void Light::toString() {}
