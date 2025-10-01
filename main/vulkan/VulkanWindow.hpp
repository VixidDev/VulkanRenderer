#pragma once

#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>
#include <cstdint>

#include "objects/VkObjects.hpp"

#include <volk/volk.h>

class VulkanDevice;
struct GLFWwindow;

class VulkanWindow {
public:
	VulkanWindow() = default;
	~VulkanWindow();

	// Move-only
	VulkanWindow(VulkanWindow const&) = delete;
	VulkanWindow& operator= (VulkanWindow const&) = delete;

	VulkanWindow(VulkanWindow&&) noexcept;
	VulkanWindow& operator= (VulkanWindow&&) noexcept;

	void setDeviceProperties(VkPhysicalDeviceProperties deviceProperties);
	void setDeviceFeatures(VkPhysicalDeviceFeatures2 deviceFeatures);

	const VkPhysicalDeviceProperties& getDeviceProperties() const;
	const VkPhysicalDeviceFeatures2& getDeviceFeatures() const;
public:
	VkInstance instance = VK_NULL_HANDLE;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	std::unique_ptr<VulkanDevice> device;

	std::uint32_t graphicsFamilyIndex = 0;
	VkQueue graphicsQueue = VK_NULL_HANDLE;
	std::uint32_t presentFamilyIndex = 0;
	VkQueue presentQueue = VK_NULL_HANDLE;

	VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

	GLFWwindow* window = nullptr;
	VkSurfaceKHR surface = VK_NULL_HANDLE;

	VkSwapchainKHR swapchain = VK_NULL_HANDLE;
	std::vector<VkImage> swapImages;
	std::vector<VkImageView> swapViews;

	VkFormat swapchainFormat;
	VkExtent2D swapchainExtent;

	std::uint32_t minImageCount = 2;
private:
	VkPhysicalDeviceProperties deviceProperties;
	VkPhysicalDeviceFeatures2 deviceFeatures;
};

struct SwapChanges {
	bool changedSize : 1;
	bool changedFormat : 1;
};

std::unique_ptr<VulkanWindow> initialiseVulkanWindow();

SwapChanges recreateSwapchain(VulkanWindow& window);

// The device selection process has changed somewhat w.r.t. the one used with VulkanContext.hpp
VkPhysicalDevice selectDevice(VkInstance instance, VkSurfaceKHR surface);
float scoreDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

std::optional<std::uint32_t> findQueueFamily(VkPhysicalDevice physicalDevice, VkQueueFlags queueFlags, VkSurfaceKHR surface = VK_NULL_HANDLE);

std::unique_ptr<VulkanDevice> createDevice(
	VulkanWindow& window,
	VkPhysicalDevice physicalDevice,
	const std::vector<std::uint32_t>& queueFamilies,
	const std::vector<const char*>& enabledDeviceExtensions = {}
);

std::tuple<VkSwapchainKHR, VkFormat, VkExtent2D, std::uint32_t> createSwapchain(
	VkPhysicalDevice physicalDevice,
	VkSurfaceKHR surface,
	VkDevice device,
	GLFWwindow* glfwWindow,
	const std::vector<std::uint32_t>& queueFamilyIndices = {},
	VkSwapchainKHR oldSwapchain = VK_NULL_HANDLE
);

std::vector<VkSurfaceFormatKHR> getSurfaceFormats(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
std::unordered_set<VkPresentModeKHR> getPresentModes(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

void getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, std::vector<VkImage>& images);
void createSwapchainImageViews(VkDevice device, VkFormat swapchainFormat, const std::vector<VkImage>& images, std::vector<VkImageView>& imageViews);

VkResult submitAndPresent(
	const VulkanWindow& window,
	std::vector<VkCommandBuffer>& cmdBuffers,
	std::vector<vk::Fence>& frameDone,
	std::vector<vk::Semaphore>& imageAvailable,
	std::vector<vk::Semaphore>& renderFinished,
	std::size_t frameIndex,
	std::uint32_t imageIndex
);