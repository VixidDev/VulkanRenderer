#pragma once

#include <vector>
#include <optional>

#include "objects/VkObjects.hpp"

class VulkanWindow;

class VulkanDevice {
public:
	VulkanDevice() noexcept = default;
	VulkanDevice(VkInstance instance, VkSurfaceKHR surface);
	~VulkanDevice();

	explicit VulkanDevice(VkDevice device, const VulkanWindow& window) noexcept;

	// Move-only
	VulkanDevice(VulkanDevice const&) = delete;
	VulkanDevice& operator= (VulkanDevice const&) = delete;
	VulkanDevice(VulkanDevice&&) noexcept;
	VulkanDevice& operator= (VulkanDevice&&) noexcept;

	VkPhysicalDevice getPhysicalDevice();
	VkDevice getDevice() const;

	const VkPhysicalDeviceProperties& getDeviceProperties() const;
	const VkPhysicalDeviceFeatures2& getDeviceFeatures() const;

	const std::vector<std::uint32_t>& getQueueFamilyIndices() const;
	std::uint32_t getGraphicsFamilyIndex();
	VkQueue getGraphicsQueue();
	std::uint32_t getPresentFamilyIndex();
	VkQueue getPresentQueue();

	VkCommandPool getCmdPool();
	VkDescriptorPool getDescPool();

	// Get sample count flag with a selected index
	VkSampleCountFlagBits getSampleCount(std::size_t index);

	std::size_t maxSampleCountIndex = 0;
	std::vector<VkSampleCountFlagBits> possibleSampleCounts = { VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_2_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_16_BIT, VK_SAMPLE_COUNT_32_BIT, VK_SAMPLE_COUNT_64_BIT };
	std::vector<const char*> msaaOptions = { "Disabled", "MSAA 2x", "MSAA 4x", "MSAA 8x", "MSAA 16x", "MSAA 32x", "MSAA 64x" };
private:
	void selectPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
	float scoreDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

	std::optional<std::uint32_t> findQueueFamily(VkPhysicalDevice physicalDevice, VkQueueFlags queueFlags, VkSurfaceKHR surface = VK_NULL_HANDLE);

	void createLogicalDevice();
	void createCommandPool();
	void createDescriptorPool();

	// Device handles
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;

	// Properties and features
	VkPhysicalDeviceProperties deviceProperties{};
	VkPhysicalDeviceFeatures2 deviceFeatures{};

	// Enabled extensions
	std::vector<const char*> enabledDevExtensions;

	// Queues and index's
	std::vector<std::uint32_t> queueFamilyIndices;
	std::uint32_t graphicsFamilyIndex = 0;
	VkQueue graphicsQueue = VK_NULL_HANDLE;
	std::uint32_t presentFamilyIndex = 0;
	VkQueue presentQueue = VK_NULL_HANDLE;

	// Pools
	VkCommandPool cmdPool = VK_NULL_HANDLE;
	VkDescriptorPool descPool = VK_NULL_HANDLE;
};