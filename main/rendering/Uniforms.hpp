#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

namespace glsl {
	struct MVPUniform {
		glm::mat4 projection = glm::mat4(1.0f);
		glm::mat4 view = glm::mat4(1.0f);
		glm::vec4 camPos{};
	};

	struct CameraPlanesUniform {
		float _far;
		float _near;
	};

	struct ProjectiveUniform {
		glm::mat4 projection = glm::mat4(1.0f);
		glm::mat4 invProjection = glm::mat4(1.0f);
	};

	struct InverseMatricesUniform {
		glm::mat4 invViewProj = glm::mat4(1.0f);
		glm::mat4 invProj = glm::mat4(1.0f);
		glm::mat4 invView = glm::mat4(1.0f);
	};

	struct SSAOUniform {
		glm::vec4 samples[32];
		float radius;
	};

	struct Light {
		glm::vec4 positionAndLightType{}; // xyz: position,  w: light type
		glm::vec4 directionAndMapIndex{}; // xyz: direction, w: shadow map index
		glm::vec4 colourAndIntensity{};   // xyz: colour,	 w: intensity
	};

	static_assert(sizeof(Light) == 48, "Light stuct must be 48 bytes!");

	// Push constant structs
	struct LightsAndEmissive {
		int numLights;
		float emissiveStrength;
		float brightnessThreshold;
		float shadowBias;
		int ssaoEnabled;
		float ssaoExp;
	};

	struct CubemapPC {
		glm::vec4 lightPos{};
		float farPlane;
	};
}