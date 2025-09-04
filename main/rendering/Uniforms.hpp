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
}