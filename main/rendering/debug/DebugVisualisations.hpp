#pragma once

#include "../vulkan/objects/VkBuffer.hpp"

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

class Renderer;

struct InstanceData {
	glm::vec3 translation;
	float scale;
	glm::vec4 colour;
};

namespace Debug {

	void renderDebugLightVolumes(Renderer* renderer, uint32_t imageIndex);

	vk::Buffer& getLightVolumesDebugBuffer(Renderer* renderer, std::vector<InstanceData>& instanceData, bool forceUpdate);

	bool shouldUpdateBuffer(std::vector<InstanceData>& instanceData);

	void destroyBuffers();

}