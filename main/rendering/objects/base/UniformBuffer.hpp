#pragma once

#include "interfaces/IUniformBuffer.hpp"
#include "../../../vulkan/objects/VkBuffer.hpp"
#include "../../../vulkan/VkUtils.hpp"

template <class T>
class UniformBuffer : public IUniformBuffer {
public:
	UniformBuffer() = default;
	UniformBuffer(VulkanAllocator* allocator, VkPipelineStageFlags stageFlags, T* uniformData)
		: allocator(allocator), stageFlags(stageFlags) 
	{
		this->buffer = vk::createBuffer(
			*this->allocator,
			sizeof(T),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			0,
			VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

		this->uniformData = uniformData;
	}

	~UniformBuffer() = default;

	void update(VkCommandBuffer cmdBuff) override {
		Utils::bufferBarrier(
			cmdBuff,
			this->buffer.buffer,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			this->stageFlags,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(cmdBuff, this->buffer.buffer, 0, sizeof(T), this->uniformData);

		Utils::bufferBarrier(
			cmdBuff,
			this->buffer.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			this->stageFlags);
	}

	VkBuffer getHandle() const override {
		return this->buffer.buffer;
	}

private:
	VulkanAllocator* allocator;
	VkPipelineStageFlags stageFlags;

	vk::Buffer buffer;
	T* uniformData;
};