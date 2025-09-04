#pragma once

#include "../../base/UniformBuffer.hpp"
#include "../../../Uniforms.hpp"

class CameraPlanesUniformBuffer : public UniformBuffer {
public:
	CameraPlanesUniformBuffer(VulkanAllocator* allocator, VkPipelineStageFlags stageFlags, glsl::CameraPlanesUniform* cameraPlanesUniform);

	void update(VkCommandBuffer cmdBuff);
private:
	glsl::CameraPlanesUniform* uniformData;
};