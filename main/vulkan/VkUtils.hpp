#pragma once

#include "objects/VkObjects.hpp"
#include "VulkanContext.hpp"
#include "../rendering/objects/base/structure/Textures.hpp"

#include <array>

struct SamplerInfo {
	VkFilter magFilter;
	VkFilter minFilter;
	VkSamplerAddressMode addressMode; 
	// In every case so far all 3 address modes are
	// the same, so just pass 1 and assume the same for the rest
	//VkSamplerAddressMode addressModeV;
	//VkSamplerAddressMode addressModeW;
	VkBool32 anisotropyEnable = VK_FALSE;
	float maxAnisotropy = 8.0f;
	VkBool32 compareEnable = 0;
	VkCompareOp compareOp = VK_COMPARE_OP_NEVER;
	VkBorderColor borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
};

namespace VkUtils {

	// Command buffers
	VkCommandBuffer createCommandBuffer(const VulkanWindow& window, VkCommandPool cmdPool);
	void beginCommandBuffer(VkCommandBuffer cmdBuff, VkCommandBufferUsageFlags usageFlags = 0);
	void endCommandBuffer(VkCommandBuffer cmdBuff);
	void endAndSubmitCommandBuffer(const VulkanWindow& window, VkCommandBuffer cmdBuff);

	// Synchronisations
	vk::Fence createFence(const VulkanWindow& window, VkFenceCreateFlags createFlags = 0);
	vk::Semaphore createSemaphore(const VulkanWindow& window);
	void waitForFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex);
	void resetFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex);

	VkResult acquireNextSwapchainImage(const VulkanWindow& window, std::vector<vk::Semaphore>& semaphores, std::size_t frameIndex, std::uint32_t& imageIndex);

	// Descriptor sets
	VkDescriptorSet createDescriptorSet(const VulkanWindow& window, VkDescriptorPool descPool, VkDescriptorSetLayout descSetLayout);

	// Samplers
	vk::Sampler createTextureSampler(const VulkanWindow& window, SamplerInfo samplerInfo);
	vk::Sampler createDefaultSampler(const VulkanWindow& window);
	vk::Sampler createShadowSampler(const VulkanWindow& window);

	// Barriers
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

	// Queries
	vk::QueryPool createQueryPool(const VulkanWindow& window, VkQueryType queryType, std::uint32_t queryCount);
	void getQueryPoolResults(
		const VulkanWindow& window, 
		vk::QueryPool& queryPool, 
		std::vector<std::size_t>& queryResults,
		std::uint32_t queryCount,
		VkQueryResultFlags resultFlags = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	);

	// Debug Labels
	void beginQueueLabel(VkQueue queue, const char* name, std::array<float, 4> colour = { 1.0f, 1.0f, 1.0f, 1.0f });
	void insertQueueLabel(VkQueue queue, const char* name, std::array<float, 4> colour = { 1.0f, 1.0f, 1.0f, 1.0f });
	void endQueueLabel(VkQueue queue);
	
	void beginCmdLabel(VkCommandBuffer cmdBuf, const char* name, std::array<float, 4> colour = { 1.0f, 1.0f, 1.0f, 1.0f });
	void insertCmdLabel(VkCommandBuffer cmdBuf, const char* name, std::array<float, 4> colour = { 1.0f, 1.0f, 1.0f, 1.0f });
	void endCmdLabel(VkCommandBuffer cmdBuf);

	void setObjectName(VkDevice device, VkObjectType type, std::uint64_t handle, const char* name);

	// Misc
	ImageLayout getFinalLayoutFromFormat(ImageFormat format);
}