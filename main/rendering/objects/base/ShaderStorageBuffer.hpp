#pragma once

#include "../../../vulkan/objects/VkObjects.hpp"
#include "../../../vulkan/objects/VkBuffer.hpp"

struct VulkanContext;

class ShaderStorageBuffer {
public:
	ShaderStorageBuffer() = default;
	ShaderStorageBuffer(VulkanContext* context);

	virtual ~ShaderStorageBuffer() = default;

	virtual void update(VkCommandBuffer cmdBuff = VK_NULL_HANDLE);

	std::uint32_t getBufferSize();
	VkBuffer getHandle();
protected:
	VulkanContext* context;

	std::size_t bufferSize = 0;

	vk::Buffer gpuBuffer;
	vk::Buffer stagingBuffer;
};