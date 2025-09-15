#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

namespace glsl {
	struct MVPUniform {
		glm::mat4 projection = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::vec4 camPos{};
	};

	struct DepthMVPUniform {
		glm::mat4 depthMVP = glm::mat4(1.0f);
	};

	struct CameraPlanesUniform {
		float _far;
		float _near;
	};

	struct Light {
		// Add padding to struct since in the shader we are using
		// std140 memory layout which forces alignment to 16 byte boundaries
		glm::vec3 position{};  float pad0;
		glm::vec3 direction{}; float pad1;
		glm::vec3 colour{};    float pad2 = 1.f;
		alignas(16) glm::ivec3 metadata{};
		// metadata.x = lightType // 0 - Point, 1 - Directional, 2 - Spot
		// metadata.y = shadowMapIndex
		// metadata.z = intensity
	};

	static_assert(sizeof(Light) == 64, "Light stuct must be 64 bytes!");
}