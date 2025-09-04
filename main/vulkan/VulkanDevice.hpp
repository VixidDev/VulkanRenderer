#pragma once

#include <volk/volk.h>

#include <vector>

#include "objects/VkObjects.hpp"

class VulkanWindow;

class VulkanDevice {
public:
	VulkanDevice() noexcept = default;
	~VulkanDevice() = default;

	explicit VulkanDevice(VkDevice device, const VulkanWindow& window) noexcept;

	// Move-only
	VulkanDevice(VulkanDevice const&) = delete;
	VulkanDevice& operator= (VulkanDevice const&) = delete;

	VulkanDevice(VulkanDevice&&) noexcept;
	VulkanDevice& operator= (VulkanDevice&&) noexcept;

	void createCommandPool(const VulkanWindow& window);
	void createDescriptorPool();

	// Get sample count flag with a selected index
	VkSampleCountFlagBits getSampleCount(std::size_t index);

	VkDevice device = VK_NULL_HANDLE;
	VkCommandPool cmdPool = VK_NULL_HANDLE;
	VkDescriptorPool descPool = VK_NULL_HANDLE;

	std::size_t maxSampleCountIndex = 0;
	std::vector<VkSampleCountFlagBits> possibleSampleCounts = { VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_2_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_32_BIT, VK_SAMPLE_COUNT_64_BIT };
	std::vector<const char*> msaaOptions = { "Disabled", "MSAA 2x", "MSAA 4x", "MSAA 8x", "MSAA 16x", "MSAA 32x", "MSAA 64x" };
};

struct SamplerInfo {
	VkFilter minFilter;
	VkFilter magFilter;
	VkSamplerAddressMode addressModeU;
	VkSamplerAddressMode addressModeV;
	VkSamplerAddressMode addressModeW;
	VkBool32 compareEnable = 0;
	VkCompareOp compareOp = VK_COMPARE_OP_NEVER;
};

VkCommandBuffer createCommandBuffer(const VulkanWindow& window);
vk::Fence createFence(const VulkanWindow& window, VkFenceCreateFlags createFlags = 0);
vk::Semaphore createSemaphore(const VulkanWindow& window);

VkDescriptorSet allocateDescriptorSet(const VulkanWindow& window, VkDescriptorPool descPool, VkDescriptorSetLayout descSetLayout);

void beginCommandBuffer(VkCommandBuffer cmdBuff, VkCommandBufferUsageFlags usageFlags = 0);
void endCommandBuffer(const VulkanWindow& window, VkCommandBuffer cmdBuff);
void endAndSubmitCommandBuffer(const VulkanWindow& window, VkCommandBuffer cmdBuff);

vk::Sampler createTextureSampler(const VulkanWindow& window, SamplerInfo samplerInfo);

void waitForFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex);
void resetFences(const VulkanWindow& window, std::vector<vk::Fence>& fences, std::size_t frameIndex);

VkResult acquireNextSwapchainImage(const VulkanWindow& window, std::vector<vk::Semaphore>& semaphores, std::size_t frameIndex, std::uint32_t& imageIndex);