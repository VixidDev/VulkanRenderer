#include "Light.hpp"

Light::Light(LightType type, glm::vec3 pos, glm::vec3 direction, glm::vec3 colour) 
	: type(type), position(pos), direction(direction), colour(colour) {}

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

void Light::toString() {}
