#pragma once

#include "interfaces/IUniformBuffer.hpp"
#include "../../../vulkan/objects/VkBuffer.hpp"
#include "../../../vulkan/VkUtils.hpp"
#include "../../../vulkan/VulkanContext.hpp"
#include "../../../vulkan/Swapchain.hpp"

template <class T>
class UniformBuffer : public IUniformBuffer {
public:
	UniformBuffer() = default;
	UniformBuffer(VulkanContext* context, VkPipelineStageFlags stageFlags, T* uniformData)
		: context(context), stageFlags(stageFlags) 
	{
		for (int i = 0; i < Swapchain::MAX_FRAMES_IN_FLIGHT; i++) {
			this->buffers.emplace_back(vk::Buffer::createBuffer(
				*this->context->allocator,
				sizeof(T),
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				0,
				VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
		}

		this->uniformData = uniformData;
	}

	~UniformBuffer() = default;

	void update(std::uint32_t frameIndex, VkCommandBuffer cmdBuff) override {
		VkUtils::bufferBarrier(
			cmdBuff,
			this->buffers[frameIndex].get(),
			/* srcAccessMask */ VK_ACCESS_UNIFORM_READ_BIT,
			/* dstAccessMask */ VK_ACCESS_TRANSFER_WRITE_BIT,
			/* srcStageMask */ this->stageFlags,
			/* dstStageMask */ VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(cmdBuff, this->buffers[frameIndex].get(), 0, sizeof(T), this->uniformData);

		VkUtils::bufferBarrier(
			cmdBuff,
			this->buffers[frameIndex].get(),
			/* srcAccessMask */ VK_ACCESS_TRANSFER_WRITE_BIT,
			/* dstAccessMask */ VK_ACCESS_UNIFORM_READ_BIT,
			/* srcStageMask */ VK_PIPELINE_STAGE_TRANSFER_BIT,
			/* dstStageMask */ this->stageFlags);
	}

	VkBuffer getHandle(std::uint32_t frameIndex) const override {
		return this->buffers[frameIndex].get();
	}

	std::vector<vk::Buffer>& getBuffers() override {
		return this->buffers;
	}

private:
	VulkanContext* context;
	VkPipelineStageFlags stageFlags;

	std::vector<vk::Buffer> buffers;
	T* uniformData;
};