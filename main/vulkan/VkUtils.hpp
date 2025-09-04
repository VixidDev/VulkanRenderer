#pragma once

#include <volk/volk.h>

#include "objects/VkObjects.hpp"
#include "VulkanContext.hpp"
#include "dbgname.hpp"

namespace Utils {

	vk::CommandPool createCommandPool(const VulkanWindow& window, VkCommandPoolCreateFlags createFlags = 0);
	VkCommandBuffer allocCommandBuffer(const VulkanWindow& window, VkCommandPool cmdPool);

	vk::Fence createFence(const VulkanWindow& window, VkFenceCreateFlags createFlags = 0);
	vk::Semaphore createSemaphore(const VulkanWindow& window);

	vk::DescriptorPool createDescriptorPool(const VulkanWindow& window, std::uint32_t maxDescriptors = 2048, std::uint32_t maxSets = 1024);

	VkDescriptorSet allocDescriptorSet(const VulkanWindow& window, VkDescriptorPool descPool, VkDescriptorSetLayout descSetLayout);

	vk::Sampler createDefaultSampler(const VulkanWindow& window);
	vk::Sampler createShadowSampler(const VulkanWindow& window);

	void bufferBarrier(
		VkCommandBuffer cmdBuff,
		VkBuffer buffer,
		VkAccessFlags srcAccessMask,
		VkAccessFlags dstAccessMask,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask,
		VkDeviceSize size = VK_WHOLE_SIZE,
		VkDeviceSize offset = 0,
		uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED
	);

	void imageBarrier(
		VkCommandBuffer cmdBuff,
		VkImage image,
		VkAccessFlags srcAccessMask,
		VkAccessFlags dstAccessMask,
		VkImageLayout srcLayout,
		VkImageLayout dstLayout,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask,
		VkImageSubresourceRange subresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		std::uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		std::uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED
	);


}