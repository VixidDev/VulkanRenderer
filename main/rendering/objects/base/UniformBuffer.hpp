#pragma once

#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkBuffer.hpp"

class VulkanAllocator;

class UniformBuffer {
public:
	UniformBuffer() = default;
	UniformBuffer(VulkanAllocator* allocator, VkPipelineStageFlags stageFlags);

	virtual ~UniformBuffer() = default;

	virtual void update(VkCommandBuffer cmdBuf);

	VkBuffer getHandle();
protected:
	VulkanAllocator* allocator;

	VkPipelineStageFlags stageFlags;

	vk::Buffer buffer;
};