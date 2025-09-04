#pragma once

#include <glm/glm.hpp>

class Light {
public:
	Light(glm::vec3 pos, bool dynamic);

	virtual void toString();

private:
	glm::vec3 pos;
	bool dynamic;
};