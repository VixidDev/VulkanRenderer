#pragma once

#include "../../base/UniformBuffer.hpp"
#include "../../../Uniforms.hpp"

class MVPUniformBuffer : public UniformBuffer {
public:
	MVPUniformBuffer(VulkanAllocator* allocator, VkPipelineStageFlags stageFlags, glsl::MVPUniform* mvpUniform);

	void update(VkCommandBuffer cmdBuff);
private:
	glsl::MVPUniform* uniformData;
};