#pragma once

#include "../../base/UniformBuffer.hpp"
#include "../../../Uniforms.hpp"

class DepthMVPUniformBuffer : public UniformBuffer {
public:
	DepthMVPUniformBuffer(VulkanAllocator* allocator, VkPipelineStageFlags stageFlags, glsl::DepthMVPUniform* mvpUniform);

	void update(VkCommandBuffer cmdBuff);
private:
	glsl::DepthMVPUniform* uniformData;
};