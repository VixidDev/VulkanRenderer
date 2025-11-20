#include "Light.hpp"

#include "magic_enum.hpp"

#include <format>

Light::Light(
	LightType type, 
	glm::vec3 pos, 
	glm::vec3 direction,
	glm::vec3 colour, 
	int intensity, 
	float innerAngle, 
	float outerAngle, 
	bool shadowCasting
) : 
	type(type), 
	position(pos), 
	direction(direction), 
	colour(colour), 
	intensity(intensity),
	innerAngle(innerAngle), 
	outerAngle(outerAngle), 
	shadowCasting(shadowCasting) {}

float Light::getRadius() const {
	return std::sqrt(this->intensity / 0.025f);
}

std::string Light::toString() const {
	return "Light[type=" + std::string(magic_enum::enum_name(type)) + ", " +
		"pos=" + std::format("x:{} y:{} z:{}, ", position[0], position[1], position[2]) +
		"direction=" + std::format("x:{} y:{} z:{}, ", direction[0], direction[1], direction[2]) +
		"colour=" + std::format("{} {} {} {}, ", colour[0], colour[1], colour[2], colour[3]) +
		"itensity=" + std::format("{}, ", intensity) +
		"innerAngle=" + std::format("{}, ", innerAngle) +
		"outerAngle=" + std::format("{}, ", outerAngle) +
		"shadowCasting=" + std::format("{}, ", shadowCasting) +
		"radius=" + std::format("{}, ", getRadius()) +
		"dirty=" + std::format("{}, ", dirty) +
		"enabled=" + std::format("{}", enabled) + "]";
}
